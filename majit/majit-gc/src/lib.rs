pub use gcreftracer::{GcTable, install_gc_table_walker};
/// GC traits and interfaces for the JIT.
///
/// The GC subsystem provides:
/// 1. Object allocation (nursery bump-pointer + old gen)
/// 2. Write barrier insertion
/// 3. GC-aware IR rewriting (NEW_* → inline nursery allocation)
/// 4. Stack maps for compiled code
///
/// Reference: rpython/memory/gc/incminimark.py, rpython/jit/backend/llsupport/gc.py
use majit_ir::{Const, ConstMap, GcRef, Op};
pub use trace::{ClassTypeLayout, TypeEntry, TypeInfo, TypeInfoLayout};

pub mod collector;
pub mod gc_sync;
pub mod gcreftracer;
pub mod header;
pub mod nursery;
pub mod oldgen;
pub mod rewrite;
pub mod shadow_stack;
pub mod trace;
pub mod weakref;

/// GC flags stored in object headers.
///
/// From incminimark.py GCFLAG_* constants.
pub mod flags {
    // incminimark.py GCFLAG_* — bit positions must match RPython exactly.
    // first_gcflag = 1 << 32; each constant below is (first_gcflag << N)
    // expressed as the unshifted bit index N.
    /// GCFLAG_TRACK_YOUNG_PTRS (bit 0)
    pub const TRACK_YOUNG_PTRS: u64 = 1 << 0;
    /// GCFLAG_NO_HEAP_PTRS (bit 1)
    pub const NO_HEAP_PTRS: u64 = 1 << 1;
    /// GCFLAG_VISITED (bit 2)
    pub const VISITED: u64 = 1 << 2;
    /// GCFLAG_HAS_SHADOW (bit 3)
    pub const HAS_SHADOW: u64 = 1 << 3;
    /// GCFLAG_FINALIZATION_ORDERING (bit 4)
    pub const FINALIZATION_ORDERING: u64 = 1 << 4;
    /// GCFLAG_EXTRA (bit 5) — reserved
    pub const EXTRA: u64 = 1 << 5;
    /// GCFLAG_HAS_CARDS (bit 6)
    pub const HAS_CARDS: u64 = 1 << 6;
    /// GCFLAG_CARDS_SET (bit 7) — MSB of the byte containing TRACK_YOUNG_PTRS.
    /// The x86 backend relies on this being -0x80 as a signed byte.
    pub const CARDS_SET: u64 = 1 << 7;
    /// GCFLAG_VISITED_RMY (bit 8)
    pub const VISITED_RMY: u64 = 1 << 8;
    /// GCFLAG_PINNED (bit 9)
    pub const PINNED: u64 = 1 << 9;
    /// GCFLAG_IGNORE_FINALIZER (bit 10)
    pub const IGNORE_FINALIZER: u64 = 1 << 10;
    /// GCFLAG_SHADOW_INITIALIZED (bit 11)
    pub const SHADOW_INITIALIZED: u64 = 1 << 11;
    /// GCFLAG_DUMMY (bit 12)
    pub const DUMMY: u64 = 1 << 12;
}

/// True when the `gc_stress` test feature is compiled in: every allocation
/// may then run a full collection inside `alloc_with_type`, so JIT fast
/// paths that bypass it (inline nursery bump) must stay disabled or the
/// stress coverage silently shrinks to non-JIT allocations.
pub fn gc_stress_enabled() -> bool {
    cfg!(feature = "gc_stress")
}

/// Write barrier descriptor — information the JIT needs to emit write barrier checks.
///
/// From rpython/jit/backend/llsupport/gc.py WriteBarrierDescr.
#[derive(Debug, Clone)]
pub struct WriteBarrierDescr {
    /// gc.py:268: GCClass.JIT_WB_IF_FLAG
    pub jit_wb_if_flag: u64,
    /// gc.py:269: extract_flag_byte(jit_wb_if_flag) → byteofs
    /// Object-relative (negative = before object start, in header).
    pub jit_wb_if_flag_byteofs: i32,
    /// gc.py:269: extract_flag_byte(jit_wb_if_flag) → singlebyte
    pub jit_wb_if_flag_singlebyte: u8,
    /// gc.py:273: GCClass.JIT_WB_CARDS_SET (0 if no card marking)
    pub jit_wb_cards_set: u64,
    /// gc.py:274: GCClass.JIT_WB_CARD_PAGE_SHIFT
    pub jit_wb_card_page_shift: u32,
    /// gc.py:275: extract_flag_byte(jit_wb_cards_set) → byteofs
    pub jit_wb_cards_set_byteofs: i32,
    /// gc.py:275: extract_flag_byte(jit_wb_cards_set) → singlebyte
    /// Must equal -0x80 (signed) per gc.py:281 assert.
    pub jit_wb_cards_set_singlebyte: i8,
}

impl WriteBarrierDescr {
    /// gc.py:285-293 extract_flag_byte: find the non-zero byte in the
    /// header-shifted flag word and return (obj_relative_byteofs, singlebyte).
    ///
    /// The returned offset is relative to the **object pointer** (not the
    /// header), matching RPython's convention where the JIT emits
    /// `load [obj + byteofs]`.  Since our header sits at `obj - GcHeader::SIZE`,
    /// the conversion is `obj_ofs = header_ofs - GcHeader::SIZE`.
    pub fn extract_flag_byte(flag: u64) -> (i32, i8) {
        let shifted = flag << crate::header::FLAG_SHIFT;
        let bytes = shifted.to_le_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            if b != 0 {
                let obj_ofs = i as i32 - crate::header::GcHeader::SIZE as i32;
                return (obj_ofs, b as i8);
            }
        }
        (0, 0)
    }

    /// Build a descriptor with correct byte offsets for the current
    /// header layout. gc.py:259-293 WriteBarrierDescr.__init__.
    pub fn for_current_gc() -> Self {
        let (if_flag_byteofs, if_flag_singlebyte) =
            Self::extract_flag_byte(flags::TRACK_YOUNG_PTRS);
        let (cards_set_byteofs, cards_set_singlebyte) = Self::extract_flag_byte(flags::CARDS_SET);
        // gc.py:280-281: the x86 backend relies on these two facts
        // to avoid one instruction in _write_barrier_fastpath.
        debug_assert_eq!(
            cards_set_byteofs, if_flag_byteofs,
            "CARDS_SET and TRACK_YOUNG_PTRS must be in the same byte"
        );
        debug_assert_eq!(
            cards_set_singlebyte, -0x80i8,
            "CARDS_SET must be the MSB of its byte (-0x80)"
        );
        WriteBarrierDescr {
            jit_wb_if_flag: flags::TRACK_YOUNG_PTRS,
            jit_wb_if_flag_byteofs: if_flag_byteofs,
            jit_wb_if_flag_singlebyte: if_flag_singlebyte as u8,
            jit_wb_cards_set: flags::CARDS_SET,
            jit_wb_card_page_shift: crate::collector::DEFAULT_CARD_PAGE_SHIFT,
            jit_wb_cards_set_byteofs: cards_set_byteofs,
            jit_wb_cards_set_singlebyte: cards_set_singlebyte,
        }
    }
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

    /// Allocate a fixed-size object with type id without triggering collection.
    ///
    /// Falls back to old-gen when nursery is full. Used for jitframe
    /// allocation where input refs on the Rust stack are not yet protected
    /// by the shadow stack (Rust stack is not traced by GC, unlike RPython
    /// stack where `lltype.malloc` can safely trigger GC).
    fn alloc_nursery_no_collect_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        let _ = type_id;
        self.alloc_nursery_no_collect(size)
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

    /// Allocate a stable-address object directly in old-gen.
    ///
    /// Used by host-side allocators (e.g. pyre-object `w_int_new`
    /// non-cached path) that return a raw pointer the caller holds on
    /// the Rust stack before it can be stored into a GC-tracked slot.
    /// Old-gen objects never move in MiniMark mark-sweep collection,
    /// so a subsequent minor collection cannot invalidate the pointer.
    ///
    /// Default implementation routes to
    /// `alloc_nursery_no_collect_typed` so backends without a
    /// distinct old-gen still compile; backends with a real old-gen
    /// override to force placement there.
    fn alloc_oldgen_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        self.alloc_nursery_no_collect_typed(type_id, size)
    }

    /// Charge `bytes` of off-heap memory pressure (e.g. a nursery object's
    /// external, GC-invisible payload such as a bignum's limb `Vec`). Mirrors
    /// RPython `rgc.add_memory_pressure`: the GC accounts memory it cannot see so
    /// collection cadence reflects true footprint. Default is a no-op so backends
    /// without a generational collector compile unchanged.
    fn charge_memory_pressure(&mut self, bytes: usize) {
        let _ = bytes;
    }

    /// Add `bytes` of `obj_addr`'s off-heap payload to the major-collection
    /// threshold's external total if the object is already old-gen, WITHOUT
    /// forcing a minor (unlike
    /// [`charge_memory_pressure`](GcAllocator::charge_memory_pressure)).
    /// Callers may pass a nursery object; generational collectors ignore it
    /// because promotion accounting will charge it later. Default is a no-op so
    /// backends without a generational collector compile unchanged.
    fn charge_oldgen_external(&mut self, obj_addr: usize, bytes: usize) {
        let _ = (obj_addr, bytes);
    }

    /// incminimark.py:1569: jit_remember_young_pointer(obj)
    /// Perform a write barrier check on `obj`.
    /// Must be called before storing a GC reference into `obj`.
    fn write_barrier(&mut self, obj: GcRef);

    /// incminimark.py:1606 jit_remember_young_pointer_from_array:
    /// Called by JIT when TRACK_YOUNG_PTRS set but CARDS_SET not.
    /// Tries to set CARDS_SET if HAS_CARDS; else generic barrier.
    fn jit_remember_young_pointer_from_array(&mut self, obj: GcRef);

    /// incminimark.py:1557 remember_young_pointer_from_array2:
    /// Full card-marking barrier with index. Called when marking a
    /// specific card after CARDS_SET is already established.
    fn remember_young_pointer_from_array2(
        &mut self,
        obj: GcRef,
        index: usize,
        card_page_shift: u32,
    );

    /// Trigger a minor (nursery) collection.
    fn collect_nursery(&mut self);

    /// Trigger a full collection.
    fn collect_full(&mut self);

    /// Trigger a non-moving old-gen-only major collection (sweep dead old-gen
    /// objects without moving the nursery). The default no-ops so a backend
    /// with no incremental old-gen lacks no method; `MiniMarkGC` overrides it.
    fn collect_oldgen_nonmoving(&mut self) {}

    /// minimark.py:1900-1915 `id_or_identityhash(gcobj)`.
    /// Return a stable address for the object that does not change
    /// across GC moves.  For nursery objects, allocates a shadow in
    /// old-gen and returns its address.  For old-gen objects, returns
    /// the object's own address.
    fn id_or_identityhash(&mut self, obj_addr: usize) -> usize {
        obj_addr
    }

    /// gc.py:268 write_barrier_descr: descriptor for the write barrier check.
    fn get_write_barrier_descr(&self) -> Option<WriteBarrierDescr> {
        None
    }

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

    /// Whether `addr` lies inside this GC's managed heap (nursery or
    /// old-gen). Used by host-side allocators to discriminate
    /// GC-allocated blocks from `std::alloc`-backed ones during the
    /// L1/L2 stepping-stone window — `dealloc_items_block` must
    /// early-return for GC-managed pointers (the GC sweeps them) and
    /// fall through to `std::alloc::dealloc` for non-managed ones.
    /// Default `false` matches stub allocators (no managed heap).
    fn is_managed_heap_object(&self, _addr: usize) -> bool {
        false
    }

    /// Current nursery free pointer.
    fn nursery_free(&self) -> *mut u8;

    /// gc.py:525-531 get_nursery_free_addr parity.
    /// Address of the mutable nursery_free field that JIT code updates.
    fn nursery_free_addr(&self) -> usize;

    /// Nursery top (end) pointer.
    fn nursery_top(&self) -> *const u8;

    /// gc.py:525-531 get_nursery_top_addr parity.
    /// Address of the mutable nursery_top field that JIT code reads.
    fn nursery_top_addr(&self) -> usize;

    /// Maximum size for nursery allocation (larger objects go to old gen directly).
    fn max_nursery_object_size(&self) -> usize;

    /// incminimark.py: card_page_indices → JIT_WB_CARD_PAGE_SHIFT.
    /// Log2 of the card page size. 0 if card marking is disabled.
    fn card_page_shift(&self) -> u32 {
        0
    }

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

    /// Register a GC type descriptor and return its type id.
    ///
    /// RPython parity: `rgc.register_custom_trace_hook(TYPE, trace_fn)`.
    fn register_type(&mut self, _info: TypeInfo) -> u32 {
        0
    }

    /// Number of registered GC types.
    fn type_count(&self) -> usize {
        0
    }

    /// Diagnostic only: `(oldgen_total_bytes, nursery_used_bytes)`.
    /// `oldgen_total_bytes` is `get_total_memory_used` (promoted + raw/large
    /// old-gen objects, NOT the nursery); `nursery_used_bytes` is the current
    /// nursery bump-pointer fill. Used to split GC-retained memory from
    /// host-heap allocations when diagnosing growth. Default `(0, 0)` for stub
    /// allocators with no byte accounting.
    fn heap_byte_stats(&self) -> (usize, usize) {
        (0, 0)
    }

    /// Diagnostic only: `(minor_collections, major_collections)` run so far.
    /// Used to attribute run time to collection cadence (e.g. old-gen churn
    /// driving repeated majors). Default `(0, 0)` for stub allocators.
    fn collection_counts(&self) -> (usize, usize) {
        (0, 0)
    }

    /// Whether a JIT inline nursery bump of `type_id` is equivalent to
    /// `alloc_with_type`'s fast path: the type registers no destructor and is
    /// not a weakref (either would need a side-list push at allocation, i.e.
    /// the slow path). Mirrors rewrite.py's malloc fast-path eligibility
    /// (types with finalizers/weakrefs keep the call). Default `false` so
    /// stub allocators keep the helper path.
    fn type_alloc_is_plain(&self, _type_id: u32) -> bool {
        false
    }

    /// Look up the fixed-object size for a registered GC type.
    ///
    /// RPython parity: this matches `cpu.bh_new(typedescr)` reading
    /// `typedescr.size` (llmodel.py / descr.py).  Default `None` keeps
    /// stub allocators (e.g. wasm/dynasm) from claiming knowledge.
    fn type_size(&self, _type_id: u32) -> Option<usize> {
        None
    }

    /// llsupport/gc.py:563 GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Maps a vtable pointer to its registered GC type id. RPython
    /// computes this arithmetically from the GC type_info_group base
    /// (gc.py:584-589); pyre's GC keeps an explicit vtable→type_id table
    /// because pyre frontends register vtables independently from the
    /// translator pipeline.
    ///
    /// Default `None` matches a GC layer with no installed mapping
    /// (e.g. dynasm/wasm stubs). The cmp_guard_class fallback panics
    /// instead of silently producing wrong code.
    fn get_typeid_from_classptr_if_gcremovetypeptr(&self, _classptr: usize) -> Option<u32> {
        None
    }

    /// Register a vtable pointer as the canonical class for a type id.
    /// Frontends call this once per type after `register_type`, mirroring
    /// how RPython's translator emits the vtable→typeid pair into the
    /// GC type_info_group.
    fn register_vtable_for_type(&mut self, _vtable: usize, _type_id: u32) {}

    /// `gctypelayout.encode_type_shapes_now` parity
    /// (gctypelayout.py:393-398): closes the type-registration phase.
    /// After freeze, `register_type` is forbidden, the
    /// `type_info_group` base address is stable, and every
    /// `is_object` type's `subclassrange_{min,max}` reflects the
    /// preorder of its inheritance chain (`assign_inheritance_ids`,
    /// rtyper/normalizecalls.py:373-389).
    ///
    /// Backends call this from `set_gc_allocator` so the embedded
    /// codegen-time pointers and bounds are immutable thereafter.
    /// Default no-op for stub allocators with no type table.
    fn freeze_types(&mut self) {}

    /// llsupport/gc.py:162 / gc.py:318 `supports_guard_gc_type` flag.
    /// `GcLLDescr_boehm` sets it to `False`; `GcLLDescr_framework` sets
    /// it to `True`. Relayed to `cpu.supports_guard_gc_type` via
    /// `llmodel.py:63`. Gates the backend's `genop_guard_guard_gc_type`,
    /// `genop_guard_guard_is_object`, and `genop_guard_guard_subclass`
    /// (x86/assembler.py:1896, 1925, 1946 `assert`) and
    /// `ConstPtrInfo.get_known_class(cpu)` at info.py:766. The default
    /// `false` matches `AbstractCPU.supports_guard_gc_type` in
    /// `rpython/jit/backend/model.py:21` and keeps backends without an
    /// installed TYPE_INFO table from emitting the guards.
    fn supports_guard_gc_type(&self) -> bool {
        false
    }

    /// llsupport/gc.py:631-642 `check_is_object` parity. Reads the
    /// typeid for `gcref` (gc.py:623-629 `get_actual_typeid`) and
    /// returns whether that type has `rclass.OBJECT` layout — i.e.
    /// whether `T_IS_RPYTHON_INSTANCE` is set in its infobits (gc.py:
    /// 631-642 walks the TYPE_INFO table to test that bit).
    ///
    /// Exposed on `cpu.check_is_object(gcptr)` via llmodel.py:541-546,
    /// which asserts `supports_guard_gc_type` before delegating. The
    /// optimizer consults this through info.py:766 inside
    /// `ConstPtrInfo.get_known_class(cpu)` to decide whether reading
    /// offset 0 of a constant gcref is safe.
    ///
    /// Returns `false` for null pointers and for backends without a
    /// type registry (matching `GcLLDescr_boehm`, which does not
    /// define `check_is_object`).
    fn check_is_object(&self, _gcref: GcRef) -> bool {
        false
    }

    /// gc/base.py:380-383 `is_valid_gc_object` tagged-immediate test:
    /// `config.taggedpointers && (addr & 1 == 1)`. Backends with no
    /// tagged-immediate support (the default) return `false`.
    fn is_tagged_immediate(&self, _addr: usize) -> bool {
        false
    }

    /// `rpython/rlib/rgc.py:229` `can_move(p)` — whether the GC object
    /// `gcref` sits at an address that may still move. "With non-moving
    /// GCs, it is always False; with moving GCs it can be True for some
    /// time, then False once the object is sure not to move." The default
    /// is `false`, matching a non-moving GC (and the no-GC case).
    fn can_move(&self, _gcref: GcRef) -> bool {
        false
    }

    /// llsupport/gc.py:592 `get_translated_info_for_typeinfo`.
    /// Returns `(type_info_group_base, shift_by, sizeof_ti)`:
    ///  * `type_info_group_base` — base address of the `TYPE_INFO` table
    ///    (`llop.gc_get_type_info_group`).
    ///  * `shift_by` — `2` on 32-bit, `0` on 64-bit (gc.py:596-599).
    ///  * `sizeof_ti` — `rffi.sizeof(GCData.TYPE_INFO)`.
    /// Called by `genop_guard_guard_is_object` (x86/assembler.py:1934)
    /// and `genop_guard_guard_subclass` (x86/assembler.py:1965).
    ///
    /// Default panics to match RPython: `GcLLDescr_boehm` does not
    /// define the method, and calling it when
    /// `supports_guard_gc_type = False` is a precondition violation.
    fn get_translated_info_for_typeinfo(&self) -> (usize, u8, usize) {
        panic!(
            "GcAllocator::get_translated_info_for_typeinfo called but the \
             GC has not installed a TYPE_INFO layout (see llsupport/gc.py:\
             592); callers must first check supports_guard_gc_type"
        )
    }

    /// llsupport/gc.py:619 `get_translated_info_for_guard_is_object`.
    /// Returns `(infobits_offset, T_IS_RPYTHON_INSTANCE_BYTE)` used by
    /// `genop_guard_guard_is_object` to locate the `infobits` byte in
    /// the `TYPE_INFO` entry and the bitmask for the
    /// `T_IS_RPYTHON_INSTANCE` flag.
    ///
    /// Default panics — same rationale as
    /// `get_translated_info_for_typeinfo`.
    fn get_translated_info_for_guard_is_object(&self) -> (usize, u8) {
        panic!(
            "GcAllocator::get_translated_info_for_guard_is_object called \
             but the GC has not installed a TYPE_INFO layout (see \
             llsupport/gc.py:619); callers must first check \
             supports_guard_gc_type"
        )
    }

    /// x86/assembler.py:1951 `cpu.subclassrange_min_offset`.
    /// Byte offset of the `subclassrange_min` field inside
    /// `rclass.CLASSTYPE`. `genop_guard_guard_subclass` uses it twice:
    /// once to read the subclassrange minimum from the object's
    /// vtable (x86/assembler.py:1956) and once to locate the same
    /// field inside a `TYPE_INFO` entry (x86/assembler.py:1968-1969).
    ///
    /// Default panics — same rationale as the other TYPE_INFO helpers.
    fn subclassrange_min_offset(&self) -> usize {
        panic!(
            "GcAllocator::subclassrange_min_offset called but the GC has \
             not installed an rclass.CLASSTYPE layout (see x86/\
             assembler.py:1951); callers must first check \
             supports_guard_gc_type"
        )
    }

    /// x86/assembler.py:1971-1974 bounds lookup at codegen time:
    ///     vtable_ptr = loc_check_against_class.getint()
    ///     vtable_ptr = rffi.cast(rclass.CLASSTYPE, vtable_ptr)
    ///     check_min = vtable_ptr.subclassrange_min
    ///     check_max = vtable_ptr.subclassrange_max
    /// Returns `(subclassrange_min, subclassrange_max)` for the class
    /// whose pointer is given, or `None` if no entry exists.
    ///
    /// Default `None` keeps backends without an installed
    /// `rclass.CLASSTYPE` layout from emitting a wrong bounds check;
    /// `genop_guard_guard_subclass` callers panic loudly when the
    /// lookup misses.
    fn subclass_range(&self, _classptr: usize) -> Option<(i64, i64)> {
        None
    }

    /// Companion to `subclass_range` keyed by typeid instead of
    /// classptr. Used by the executor's `GuardSubclass` arm after it
    /// resolves `value.typeptr` via `get_actual_typeid`
    /// (llgraph/runner.py:1271-1281). Default `None`.
    fn typeid_subclass_range(&self, _typeid: u32) -> Option<(i64, i64)> {
        None
    }

    /// gc.py:624-629 `get_actual_typeid` parity. Reads the typeid
    /// from the GC header half-word for managed objects, or resolves
    /// the foreign object's classptr through `vtable_to_type_id` for
    /// backends that register a seam (e.g. pyre's PyObject layout).
    /// Default `None` for stubs without a type registry.
    fn get_actual_typeid(&self, _gcref: GcRef) -> Option<u32> {
        None
    }

    /// Companion to `check_is_object` keyed by typeid. Returns
    /// whether the typeid carries `T_IS_RPYTHON_INSTANCE` in its
    /// TYPE_INFO entry (gctypelayout.py:642). Default `None`.
    fn typeid_is_object(&self, _typeid: u32) -> Option<bool> {
        None
    }
}

/// Forwarding handle to the process-global GC singleton via `gc_sync`.
///
/// Every method routes through `gc_sync::gc_op` (mutex-guarded) or
/// `gc_sync::gc_query`. No raw pointer — synchronisation is structural.
/// Per-thread backend TLS stores `Box<GcHandle>` as `Box<dyn GcAllocator>`,
/// so the ~45 trampoline functions per backend keep their existing
/// `RefCell<Option<Box<dyn GcAllocator>>>` access pattern unchanged.
///
/// # Thread safety
///
/// All `&mut self` methods acquire `gc_sync::gc_mutex` internally.
/// Concurrent calls from different threads serialise correctly.
/// Collection uses STW safepoint protocol (`gc_sync::request_stw`).
/// See gh#396 for the full free-threading GC design.
pub struct GcHandle;

// GcHandle is a zero-size marker; Send is trivially safe.
unsafe impl Send for GcHandle {}

impl GcAllocator for GcHandle {
    fn alloc_nursery(&mut self, size: usize) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_nursery(size))
    }
    fn alloc_nursery_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_nursery_typed(type_id, size))
    }
    fn alloc_nursery_no_collect(&mut self, size: usize) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_nursery_no_collect(size))
    }
    fn alloc_varsize(&mut self, base_size: usize, item_size: usize, length: usize) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_varsize(base_size, item_size, length))
    }
    fn alloc_varsize_typed(
        &mut self,
        type_id: u32,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_varsize_typed(type_id, base_size, item_size, length))
    }
    fn alloc_nursery_no_collect_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_nursery_no_collect_typed(type_id, size))
    }
    fn alloc_varsize_no_collect(
        &mut self,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_varsize_no_collect(base_size, item_size, length))
    }
    fn alloc_oldgen_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_oldgen_typed(type_id, size))
    }
    fn charge_memory_pressure(&mut self, bytes: usize) {
        gc_sync::gc_op(|gc| gc.charge_memory_pressure(bytes))
    }
    fn charge_oldgen_external(&mut self, obj_addr: usize, bytes: usize) {
        gc_sync::gc_op(|gc| gc.charge_oldgen_external(obj_addr, bytes))
    }
    fn write_barrier(&mut self, obj: GcRef) {
        gc_sync::gc_op(|gc| gc.write_barrier(obj))
    }
    fn jit_remember_young_pointer_from_array(&mut self, obj: GcRef) {
        gc_sync::gc_op(|gc| gc.jit_remember_young_pointer_from_array(obj))
    }
    fn remember_young_pointer_from_array2(
        &mut self,
        obj: GcRef,
        index: usize,
        card_page_shift: u32,
    ) {
        gc_sync::gc_op(|gc| gc.remember_young_pointer_from_array2(obj, index, card_page_shift))
    }
    fn collect_nursery(&mut self) {
        gc_sync::gc_op(|gc| gc.collect_nursery())
    }
    fn collect_full(&mut self) {
        gc_sync::gc_op(|gc| gc.collect_full())
    }
    fn collect_oldgen_nonmoving(&mut self) {
        gc_sync::gc_op(|gc| gc.collect_oldgen_nonmoving())
    }
    fn id_or_identityhash(&mut self, obj_addr: usize) -> usize {
        gc_sync::gc_op(|gc| gc.id_or_identityhash(obj_addr))
    }
    fn get_write_barrier_descr(&self) -> Option<WriteBarrierDescr> {
        gc_sync::gc_query(|gc| gc.get_write_barrier_descr())
    }
    unsafe fn add_root(&mut self, root: *mut GcRef) {
        gc_sync::gc_op(|gc| unsafe { gc.add_root(root) })
    }
    fn remove_root(&mut self, root: *mut GcRef) {
        gc_sync::gc_op(|gc| gc.remove_root(root))
    }
    fn is_managed_heap_object(&self, addr: usize) -> bool {
        gc_sync::gc_query(|gc| gc.is_managed_heap_object(addr))
    }
    fn nursery_free(&self) -> *mut u8 {
        gc_sync::gc_query(|gc| gc.nursery_free())
    }
    fn nursery_free_addr(&self) -> usize {
        gc_sync::gc_query(|gc| gc.nursery_free_addr())
    }
    fn nursery_top(&self) -> *const u8 {
        gc_sync::gc_query(|gc| gc.nursery_top())
    }
    fn nursery_top_addr(&self) -> usize {
        gc_sync::gc_query(|gc| gc.nursery_top_addr())
    }
    fn max_nursery_object_size(&self) -> usize {
        gc_sync::gc_query(|gc| gc.max_nursery_object_size())
    }
    fn card_page_shift(&self) -> u32 {
        gc_sync::gc_query(|gc| gc.card_page_shift())
    }
    fn jit_remember_young_pointer(&mut self, obj: GcRef) {
        gc_sync::gc_op(|gc| gc.jit_remember_young_pointer(obj))
    }
    fn can_optimize_cond_call(&self) -> bool {
        gc_sync::gc_query(|gc| gc.can_optimize_cond_call())
    }
    fn gc_step(&mut self) -> bool {
        gc_sync::gc_op(|gc| gc.gc_step())
    }
    fn jit_free(&mut self, code_ptr: usize, size: usize) {
        gc_sync::gc_op(|gc| gc.jit_free(code_ptr, size))
    }
    fn pin(&mut self, obj: GcRef) -> bool {
        gc_sync::gc_op(|gc| gc.pin(obj))
    }
    fn unpin(&mut self, obj: GcRef) {
        gc_sync::gc_op(|gc| gc.unpin(obj))
    }
    fn is_pinned(&self, obj: GcRef) -> bool {
        gc_sync::gc_query(|gc| gc.is_pinned(obj))
    }
    fn register_type(&mut self, info: trace::TypeInfo) -> u32 {
        gc_sync::gc_op(|gc| gc.register_type(info))
    }
    fn type_count(&self) -> usize {
        gc_sync::gc_query(|gc| gc.type_count())
    }
    fn heap_byte_stats(&self) -> (usize, usize) {
        gc_sync::gc_query(|gc| gc.heap_byte_stats())
    }
    fn collection_counts(&self) -> (usize, usize) {
        gc_sync::gc_query(|gc| gc.collection_counts())
    }
    fn type_alloc_is_plain(&self, type_id: u32) -> bool {
        gc_sync::gc_query(|gc| gc.type_alloc_is_plain(type_id))
    }
    fn type_size(&self, type_id: u32) -> Option<usize> {
        gc_sync::gc_query(|gc| gc.type_size(type_id))
    }
    fn get_typeid_from_classptr_if_gcremovetypeptr(&self, classptr: usize) -> Option<u32> {
        gc_sync::gc_query(|gc| gc.get_typeid_from_classptr_if_gcremovetypeptr(classptr))
    }
    fn register_vtable_for_type(&mut self, vtable: usize, type_id: u32) {
        gc_sync::gc_op(|gc| gc.register_vtable_for_type(vtable, type_id))
    }
    fn freeze_types(&mut self) {
        gc_sync::gc_op(|gc| gc.freeze_types())
    }
    fn supports_guard_gc_type(&self) -> bool {
        gc_sync::gc_query(|gc| gc.supports_guard_gc_type())
    }
    fn check_is_object(&self, gcref: GcRef) -> bool {
        gc_sync::gc_query(|gc| gc.check_is_object(gcref))
    }
    fn is_tagged_immediate(&self, addr: usize) -> bool {
        gc_sync::gc_query(|gc| gc.is_tagged_immediate(addr))
    }
    fn can_move(&self, gcref: GcRef) -> bool {
        gc_sync::gc_query(|gc| gc.can_move(gcref))
    }
    fn get_translated_info_for_typeinfo(&self) -> (usize, u8, usize) {
        gc_sync::gc_query(|gc| gc.get_translated_info_for_typeinfo())
    }
    fn get_translated_info_for_guard_is_object(&self) -> (usize, u8) {
        gc_sync::gc_query(|gc| gc.get_translated_info_for_guard_is_object())
    }
    fn subclassrange_min_offset(&self) -> usize {
        gc_sync::gc_query(|gc| gc.subclassrange_min_offset())
    }
    fn subclass_range(&self, classptr: usize) -> Option<(i64, i64)> {
        gc_sync::gc_query(|gc| gc.subclass_range(classptr))
    }
    fn typeid_subclass_range(&self, typeid: u32) -> Option<(i64, i64)> {
        gc_sync::gc_query(|gc| gc.typeid_subclass_range(typeid))
    }
    fn get_actual_typeid(&self, gcref: GcRef) -> Option<u32> {
        gc_sync::gc_query(|gc| gc.get_actual_typeid(gcref))
    }
    fn typeid_is_object(&self, typeid: u32) -> Option<bool> {
        gc_sync::gc_query(|gc| gc.typeid_is_object(typeid))
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
    /// Rewrite with access to the constant pool.
    /// Returns (rewritten ops, merged constants, gc_table gcrefs). Each
    /// `Const` box carries its own type via `Const::get_type`, so a separate
    /// type side-table is no longer threaded through the return. The third
    /// element is the per-loop reference-constant list collected by
    /// `remove_constptr` (rewrite.py:1033-1043 `gcrefs_output_list`); the
    /// backend builds a `GcTable` from it and bakes its base address into the
    /// `LoadFromGcTable` loads.
    ///
    /// The default impl forwards to `rewrite_for_gc` and preserves the
    /// caller's constants verbatim. `rewrite_for_gc` may leave `Const*`
    /// operands untouched, so returning an empty map would silently strand
    /// downstream readers that resolve `ConstInt.raw()` against this table.
    fn rewrite_for_gc_with_constants(
        &self,
        ops: &[Op],
        constants: &ConstMap<Const>,
    ) -> (Vec<Op>, ConstMap<Const>, Vec<GcRef>) {
        (self.rewrite_for_gc(ops), constants.clone(), Vec::new())
    }
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

// ─────────────────────────────────────────────────────────────────────
// Thread-local active GC allocator hook
// ─────────────────────────────────────────────────────────────────────
//
// The metainterp / optimizer layer needs a backend-agnostic way to query
// the current CPU's GC type registry (llmodel.py:541-546
// `cpu.check_is_object(gcptr)`). In RPython the optimizer reaches it via
// `self.optimizer.cpu`, which holds a reference to the backend-provided
// CPU object. majit has no such field; instead the live backends register
// a callback here that the metainterp can invoke without taking a
// backend dependency.

use std::cell::Cell;

/// Thread-local callback that answers `cpu.check_is_object(gcptr)` for
/// the currently active backend. Set by the backend when it installs a
/// GC runtime for the executing thread; cleared when the runtime is
/// unregistered.
pub type CheckIsObjectFn = fn(GcRef) -> bool;

/// Thread-local callback that answers the collector's tagged-immediate
/// test (gc/base.py:380-383 `is_valid_gc_object`) for the currently
/// active backend's GC: `config.taggedpointers && (addr & 1 == 1)`.
/// Lets a backend-agnostic caller decide that an odd-valued constant
/// address is an unboxed immediate rather than a heap object, without
/// reading an object header at offset 0.
pub type IsTaggedImmediateFn = fn(usize) -> bool;

/// Thread-local callback that answers `gc_ll_descr.get_actual_typeid`
/// (gc.py:624-629) for the currently active backend. Returns the
/// `rffi.cast(HDRPTR, gcptr).tid` half-word for managed objects and
/// `vtable_to_type_id` for the foreign-object seam pyre uses. Paired
/// with `ACTIVE_CHECK_IS_OBJECT`; both are installed together so the
/// metainterp's guard interpretation stays consistent with the GC's
/// runtime layout assumptions.
pub type GetActualTypeidFn = fn(GcRef) -> Option<u32>;

/// Thread-local callback that answers the codegen-time bounds lookup
/// `rffi.cast(rclass.CLASSTYPE, vtable_ptr).subclassrange_{min,max}`
/// from x86/assembler.py:1971-1974. Used by the executor's
/// `GuardSubclass` arm to evaluate bridges interpretively.
pub type SubclassRangeFn = fn(classptr: usize) -> Option<(i64, i64)>;

/// Thread-local callback that answers the `value.typeptr
/// .subclassrange_min/max` lookup from llgraph/runner.py:1271-1281
/// directly by typeid. The backend installs this alongside
/// `subclass_range` so the executor can recover the object side of
/// `execute_guard_subclass` without going through a vtable pointer —
/// managed objects carry only a typeid in their GC header, and the
/// TYPE_INFO table already stores the preorder bounds in its paired
/// `CLASSTYPE` entry (gctypelayout.py:359-374).
pub type TypeidSubclassRangeFn = fn(typeid: u32) -> Option<(i64, i64)>;

/// Thread-local callback that answers `rclass.OBJECT`-layout queries
/// by typeid — "does this typeid carry `T_IS_RPYTHON_INSTANCE` in its
/// TYPE_INFO entry" (gctypelayout.py:642). The executor's
/// `GuardIsObject` arm calls this after resolving the object's typeid
/// via the `get_actual_typeid` seam, avoiding a second indirection
/// through `check_is_object` (which would re-resolve the typeid).
pub type TypeidIsObjectFn = fn(typeid: u32) -> Option<bool>;
pub type ExtraRootWalkerFn = fn(&mut dyn FnMut(&mut GcRef));

/// Thread-local callback that answers `rgc.can_move(gcref)`
/// (rpython/rlib/rgc.py:229) for the currently active backend's GC. The
/// const-baking site (`x86/regalloc.py:58-61 convert_to_imm`) consults
/// this before baking a `ConstPtr` immediate.
pub type CanMoveFn = fn(GcRef) -> bool;

thread_local! {
    static ACTIVE_CHECK_IS_OBJECT: Cell<Option<CheckIsObjectFn>> = const { Cell::new(None) };
    static ACTIVE_IS_TAGGED_IMMEDIATE: Cell<Option<IsTaggedImmediateFn>> = const { Cell::new(None) };
    static ACTIVE_GET_ACTUAL_TYPEID: Cell<Option<GetActualTypeidFn>> = const { Cell::new(None) };
    static ACTIVE_SUBCLASS_RANGE: Cell<Option<SubclassRangeFn>> = const { Cell::new(None) };
    static ACTIVE_TYPEID_SUBCLASS_RANGE: Cell<Option<TypeidSubclassRangeFn>> = const { Cell::new(None) };
    static ACTIVE_TYPEID_IS_OBJECT: Cell<Option<TypeidIsObjectFn>> = const { Cell::new(None) };
    static ACTIVE_CAN_MOVE: Cell<Option<CanMoveFn>> = const { Cell::new(None) };
    static ACTIVE_SUPPORTS_GUARD_GC_TYPE: Cell<bool> = const { Cell::new(false) };
    static ACTIVE_EXTRA_ROOT_WALKER: Cell<Option<ExtraRootWalkerFn>> = const { Cell::new(None) };
}

/// Bundle of callbacks the metainterp / executor can reach through
/// thread-locals. Mirrors the fan-out of methods RPython's optimizer
/// and blackhole reach via `self.cpu` / `self.cpu.gc_ll_descr`; majit
/// installs them together so a backend swap is a single call.
#[derive(Clone, Copy, Default)]
pub struct ActiveGcGuardHooks {
    pub check_is_object: Option<CheckIsObjectFn>,
    pub is_tagged_immediate: Option<IsTaggedImmediateFn>,
    pub get_actual_typeid: Option<GetActualTypeidFn>,
    pub subclass_range: Option<SubclassRangeFn>,
    pub typeid_subclass_range: Option<TypeidSubclassRangeFn>,
    pub typeid_is_object: Option<TypeidIsObjectFn>,
    pub can_move: Option<CanMoveFn>,
    pub supports_guard_gc_type: bool,
}

/// Install the active backend's GC-guard callbacks on this thread.
/// Called by backends when they enter a JIT region. Pass a default
/// `ActiveGcGuardHooks` with every field set to `None` / `false` to
/// clear. Mirrors how RPython's `cpu` field lets the optimizer and
/// executor reach `cpu.check_is_object`, `gc_ll_descr
/// .get_actual_typeid`, and the codegen-time bounds lookup; majit
/// bundles them here so a backend install is a single call.
pub fn set_active_gc_guard_hooks(hooks: ActiveGcGuardHooks) {
    ACTIVE_CHECK_IS_OBJECT.with(|c| c.set(hooks.check_is_object));
    ACTIVE_IS_TAGGED_IMMEDIATE.with(|c| c.set(hooks.is_tagged_immediate));
    ACTIVE_GET_ACTUAL_TYPEID.with(|c| c.set(hooks.get_actual_typeid));
    ACTIVE_SUBCLASS_RANGE.with(|c| c.set(hooks.subclass_range));
    ACTIVE_TYPEID_SUBCLASS_RANGE.with(|c| c.set(hooks.typeid_subclass_range));
    ACTIVE_TYPEID_IS_OBJECT.with(|c| c.set(hooks.typeid_is_object));
    ACTIVE_CAN_MOVE.with(|c| c.set(hooks.can_move));
    ACTIVE_SUPPORTS_GUARD_GC_TYPE.with(|c| c.set(hooks.supports_guard_gc_type));
}

/// Install a thread-local callback that exposes non-shadow-stack roots
/// owned by the embedding runtime.
pub fn set_active_extra_root_walker(walker: Option<ExtraRootWalkerFn>) {
    ACTIVE_EXTRA_ROOT_WALKER.with(|c| c.set(walker));
}

/// Walk the active runtime's extra GC roots.
pub fn walk_active_extra_roots(visitor: &mut dyn FnMut(&mut GcRef)) {
    ACTIVE_EXTRA_ROOT_WALKER.with(|c| {
        if let Some(f) = c.get() {
            f(visitor);
        }
    });
}

/// llmodel.py:541-546 `cpu.check_is_object(gcptr)` shim. Returns whether
/// `gcref` is a `T_IS_RPYTHON_INSTANCE` (has `typeptr` at offset 0). When
/// no backend has installed a callback on this thread, returns `false`.
pub fn check_is_object(gcref: GcRef) -> bool {
    if gcref.is_null() {
        return false;
    }
    ACTIVE_CHECK_IS_OBJECT.with(|c| match c.get() {
        Some(f) => f(gcref),
        None => false,
    })
}

/// gc/base.py:380-383 `is_valid_gc_object` tagged-immediate test shim.
/// Delegates to the active backend's installed callback, which reads its
/// GC's `config.taggedpointers`. Returns `false` for null and when no
/// backend is installed — same absent-backend semantics as
/// `check_is_object`, so flag-off / no-GC paths are unaffected.
pub fn is_tagged_immediate(addr: usize) -> bool {
    ACTIVE_IS_TAGGED_IMMEDIATE.with(|c| match c.get() {
        Some(f) => f(addr),
        None => false,
    })
}

/// Whether the active backend's GC has `config.taggedpointers` enabled
/// (translationoption.py:185). The installed `is_tagged_immediate` callback
/// answers `config.taggedpointers && (addr & 1 == 1)`, so probing it with an
/// odd sentinel address isolates the config flag without a live pointer.
/// Returns `false` when no backend is installed — same absent-backend
/// semantics as [`is_tagged_immediate`], so flag-off paths are unaffected.
pub fn taggedpointers_enabled() -> bool {
    is_tagged_immediate(1)
}

/// gc.py:624-629 `gc_ll_descr.get_actual_typeid(gcptr)` shim.
/// Delegates to the active backend's installed callback; returns
/// `None` when no backend is installed, which mirrors
/// `llgraph/runner.py:1263-1269` skip semantics (the interpretive
/// guard treats an unresolved object as passing).
pub fn get_actual_typeid(gcref: GcRef) -> Option<u32> {
    if gcref.is_null() {
        return None;
    }
    ACTIVE_GET_ACTUAL_TYPEID.with(|c| match c.get() {
        Some(f) => f(gcref),
        None => None,
    })
}

/// `rgc.can_move(gcref)` shim (rpython/rlib/rgc.py:229). Delegates to the
/// active backend's installed callback. Returns `false` for null pointers
/// and when no backend is installed — i.e. a non-moving / absent GC, where
/// every object address is stable (rgc.py:231 "with non-moving GCs, it is
/// always False").
pub fn can_move(gcref: GcRef) -> bool {
    if gcref.is_null() {
        return false;
    }
    ACTIVE_CAN_MOVE.with(|c| match c.get() {
        Some(f) => f(gcref),
        None => false,
    })
}

/// x86/assembler.py:1971-1974 codegen-time bounds lookup shim used by
/// the interpretive `GuardSubclass`. Returns
/// `(subclassrange_min, subclassrange_max)` for the class whose vtable
/// pointer is given, or `None` when no backend is installed.
pub fn subclass_range(classptr: usize) -> Option<(i64, i64)> {
    ACTIVE_SUBCLASS_RANGE.with(|c| c.get().and_then(|f| f(classptr)))
}

/// Companion to `subclass_range` keyed by typeid instead of classptr.
/// Resolves `value.typeptr.subclassrange_min/max` from
/// llgraph/runner.py:1271-1281 when the executor only has a typeid in
/// hand (e.g. after calling `get_actual_typeid` on an object whose
/// classptr is known only to the GC). Returns `None` when no backend
/// is installed.
pub fn typeid_subclass_range(typeid: u32) -> Option<(i64, i64)> {
    ACTIVE_TYPEID_SUBCLASS_RANGE.with(|c| c.get().and_then(|f| f(typeid)))
}

/// Companion to `check_is_object` keyed by typeid. Called by the
/// executor's `GuardIsObject` arm after resolving the object to a
/// typeid via `get_actual_typeid`. Returns `None` when no backend is
/// installed.
pub fn typeid_is_object(typeid: u32) -> Option<bool> {
    ACTIVE_TYPEID_IS_OBJECT.with(|c| c.get().and_then(|f| f(typeid)))
}

/// llmodel.py:63 `supports_guard_gc_type` shim. Mirrors the active
/// backend's capability flag. `false` when no backend has been installed.
pub fn supports_guard_gc_type() -> bool {
    ACTIVE_SUPPORTS_GUARD_GC_TYPE.with(|c| c.get())
}

// ── Host-side nursery allocation hook ───────────────────────────────
//
// Separate from `ActiveGcGuardHooks` because allocation is not a
// guard-time concern. The backend installs one function pointer here
// so host-side allocators (pyre-object's `w_int_new`, `w_float_new`,
// …) can route through the real GC without taking a backend-specific
// dependency. Mirrors how RPython host code reaches `gc.malloc(TYPE)`
// through the global GC instance.

/// Thread-local callback that performs a nursery allocation for the
/// currently active backend. The callback returns `GcRef(0)` (i.e.
/// null) on allocation failure so callers can fall back to a
/// non-GC allocator.
pub type AllocNurseryTypedFn = fn(type_id: u32, payload_size: usize) -> GcRef;

thread_local! {
    static ACTIVE_ALLOC_NURSERY_TYPED: Cell<Option<AllocNurseryTypedFn>> =
        const { Cell::new(None) };
}

/// Install the active backend's nursery allocator callback. Pass
/// `None` to clear.
pub fn set_active_alloc_nursery_typed(hook: Option<AllocNurseryTypedFn>) {
    ACTIVE_ALLOC_NURSERY_TYPED.with(|c| c.set(hook));
}

/// Allocate through the active backend's GC. Returns `GcRef(0)` when
/// no backend is installed on this thread (callers treat this as a
/// null pointer and fall back to their non-GC path).
pub fn alloc_nursery_typed(type_id: u32, payload_size: usize) -> GcRef {
    ACTIVE_ALLOC_NURSERY_TYPED.with(|c| match c.get() {
        Some(f) => f(type_id, payload_size),
        None => GcRef(0),
    })
}

/// Thread-local callback that performs a stable-address old-gen
/// allocation for the currently active backend. Used by host-side
/// allocators whose callers hold the returned pointer on the Rust
/// stack without registering it as a GC root. MiniMark's
/// old-gen is mark-sweep (non-moving), so a subsequent minor
/// collection cannot invalidate the pointer. The callback returns
/// `GcRef(0)` on allocation failure.
pub type AllocOldgenTypedFn = fn(type_id: u32, payload_size: usize) -> GcRef;

thread_local! {
    static ACTIVE_ALLOC_OLDGEN_TYPED: Cell<Option<AllocOldgenTypedFn>> =
        const { Cell::new(None) };
}

/// Install the active backend's old-gen allocator callback. Pass
/// `None` to clear.
pub fn set_active_alloc_oldgen_typed(hook: Option<AllocOldgenTypedFn>) {
    ACTIVE_ALLOC_OLDGEN_TYPED.with(|c| c.set(hook));
}

/// Allocate a stable-address (old-gen) object through the active
/// backend's GC. Returns `GcRef(0)` when no backend is installed on
/// this thread.
pub fn alloc_oldgen_typed(type_id: u32, payload_size: usize) -> GcRef {
    ACTIVE_ALLOC_OLDGEN_TYPED.with(|c| match c.get() {
        Some(f) => f(type_id, payload_size),
        None => GcRef(0),
    })
}

/// Thread-local callback for a *collecting* nursery allocation — unlike
/// [`alloc_nursery_typed`] (which the backends install as the no-collect
/// variant), this one runs a minor collection when the nursery is full instead
/// of spilling to old-gen. Only safe for callers that hold no unrooted GC
/// pointer across the allocation AND run at a JIT safepoint whose gcmap roots
/// the live set (e.g. the elidable bigint payload helpers, invoked from a
/// gcmap-carrying residual CallR). Returns `GcRef(0)` when no backend installed.
pub type AllocNurseryCollectingTypedFn = fn(type_id: u32, payload_size: usize) -> GcRef;

thread_local! {
    static ACTIVE_ALLOC_NURSERY_COLLECTING_TYPED: Cell<Option<AllocNurseryCollectingTypedFn>> =
        const { Cell::new(None) };
}

/// Install the active backend's collecting-nursery allocator callback. Pass
/// `None` to clear. Backends that do not install one leave callers to fall back
/// to the no-collect path.
pub fn set_active_alloc_nursery_collecting_typed(hook: Option<AllocNurseryCollectingTypedFn>) {
    ACTIVE_ALLOC_NURSERY_COLLECTING_TYPED.with(|c| c.set(hook));
}

/// Allocate through the active backend's collecting nursery allocator. Returns
/// `GcRef(0)` when no backend (or no collecting hook) is installed on this
/// thread (callers treat this as null and fall back to the no-collect path).
pub fn alloc_nursery_collecting_typed(type_id: u32, payload_size: usize) -> GcRef {
    ACTIVE_ALLOC_NURSERY_COLLECTING_TYPED.with(|c| match c.get() {
        Some(f) => f(type_id, payload_size),
        None => GcRef(0),
    })
}

/// Thread-local callback that charges off-heap memory pressure on the active
/// backend's GC (`GcAllocator::charge_memory_pressure`). Used by host-side
/// allocators of GC objects whose payload includes external, GC-invisible memory
/// (the bignum limb `Vec`). Returns silently when no backend is installed.
pub type ChargeMemoryPressureFn = fn(bytes: usize);

thread_local! {
    static ACTIVE_CHARGE_MEMORY_PRESSURE: Cell<Option<ChargeMemoryPressureFn>> =
        const { Cell::new(None) };
}

/// Install the active backend's memory-pressure callback. Pass `None` to clear.
pub fn set_active_charge_memory_pressure(hook: Option<ChargeMemoryPressureFn>) {
    ACTIVE_CHARGE_MEMORY_PRESSURE.with(|c| c.set(hook));
}

/// Charge `bytes` of off-heap memory pressure on the active backend's GC. No-op
/// when no backend is installed on this thread.
pub fn charge_memory_pressure(bytes: usize) {
    ACTIVE_CHARGE_MEMORY_PRESSURE.with(|c| {
        if let Some(f) = c.get() {
            f(bytes);
        }
    })
}

/// Thread-local callback that charges an object's off-heap payload against the
/// active backend's major threshold (`GcAllocator::charge_oldgen_external`) when
/// the object is old-gen, without forcing a minor. Used after initializing GC
/// objects whose payload includes external, GC-invisible memory (the bignum limb
/// `Vec`). Returns silently when no backend is installed.
pub type ChargeOldgenExternalFn = fn(obj_addr: usize, bytes: usize);

thread_local! {
    static ACTIVE_CHARGE_OLDGEN_EXTERNAL: Cell<Option<ChargeOldgenExternalFn>> =
        const { Cell::new(None) };
}

/// Install the active backend's old-gen external-byte callback. Pass `None` to clear.
pub fn set_active_charge_oldgen_external(hook: Option<ChargeOldgenExternalFn>) {
    ACTIVE_CHARGE_OLDGEN_EXTERNAL.with(|c| c.set(hook));
}

/// Charge `bytes` of `obj_addr`'s off-heap payload on the active backend's GC
/// when the object is old-gen. No-op when no backend is installed on this thread.
pub fn charge_oldgen_external(obj_addr: usize, bytes: usize) {
    ACTIVE_CHARGE_OLDGEN_EXTERNAL.with(|c| {
        if let Some(f) = c.get() {
            f(obj_addr, bytes);
        }
    })
}

/// Thread-local callback that runs a full mark-sweep collection cycle
/// on the active backend's GC (`GcAllocator::collect_full`). Used by
/// `pypy/module/gc/interp_gc.py:7-26 collect` ports — i.e. user-level
/// `gc.collect()` reaches the live GC through this trampoline. Returns
/// silently when no backend is installed on this thread (callers treat
/// it as a no-op).
pub type CollectFullFn = fn();

thread_local! {
    static ACTIVE_COLLECT_FULL: Cell<Option<CollectFullFn>> = const { Cell::new(None) };
}

/// Install the active backend's full-collection trampoline. Pass
/// `None` to clear.
pub fn set_active_collect_full(hook: Option<CollectFullFn>) {
    ACTIVE_COLLECT_FULL.with(|c| c.set(hook));
}

/// Trigger a full mark-sweep collection on the active backend's GC.
/// No-op when no backend is installed on this thread.
pub fn collect_full() {
    ACTIVE_COLLECT_FULL.with(|c| {
        if let Some(f) = c.get() {
            f();
        }
    });
}

/// Thread-local callback running a non-moving old-gen-only major collection
/// (`GcAllocator::collect_oldgen_nonmoving`). The interpreter GC safepoint
/// reaches it to reclaim stable-allocated interp int/float without moving the
/// nursery — so it can fire under an active JIT (nursery non-empty), unlike
/// the moving `collect_full`. No-op when no backend is installed.
pub type CollectOldgenFn = fn();

thread_local! {
    static ACTIVE_COLLECT_OLDGEN: Cell<Option<CollectOldgenFn>> = const { Cell::new(None) };
}

/// Install the active backend's non-moving-major trampoline. Pass `None` to
/// clear.
pub fn set_active_collect_oldgen(hook: Option<CollectOldgenFn>) {
    ACTIVE_COLLECT_OLDGEN.with(|c| c.set(hook));
}

/// Trigger a non-moving old-gen-only major collection on the active backend's
/// GC. No-op when no backend is installed on this thread.
pub fn collect_oldgen_nonmoving() {
    ACTIVE_COLLECT_OLDGEN.with(|c| {
        if let Some(f) = c.get() {
            f();
        }
    });
}

/// Thread-local callback reporting the active GC's `heap_byte_stats`
/// (`(oldgen_total, nursery_used)`). Lets the interpreter safepoint
/// (`pyre_object::gc_interp`) gate a collection on an empty nursery,
/// where the embedded minor cycle moves nothing and is therefore safe
/// even without a shadowstack pass over Rust-stack temporaries.
pub type HeapStatsFn = fn() -> (usize, usize);

thread_local! {
    static ACTIVE_HEAP_STATS: Cell<Option<HeapStatsFn>> = const { Cell::new(None) };
}

/// Install the active backend's `heap_byte_stats` trampoline.
pub fn set_active_heap_stats(hook: Option<HeapStatsFn>) {
    ACTIVE_HEAP_STATS.with(|c| c.set(hook));
}

/// Report `(oldgen_total, nursery_used)` from the active backend's GC.
/// `(0, 0)` when no backend is installed on this thread.
pub fn active_heap_stats() -> (usize, usize) {
    ACTIVE_HEAP_STATS.with(|c| match c.get() {
        Some(f) => f(),
        None => (0, 0),
    })
}

/// Whether the JIT-frame shadow stack is empty — i.e. no compiled trace
/// is suspended on this thread. The interpreter GC safepoint only
/// collects when this holds: a suspended jitframe's gcmap describes its
/// own suspension PC, and a collection driven from the nested interpreter
/// (not from compiled code at a real safepoint) can mis-root it. The
/// JIT's own nursery-full collections are safe; this gate keeps the
/// interpreter-driven one out of the trace-suspended window.
pub fn jitframe_shadow_stack_empty() -> bool {
    shadow_stack::jf_top_ptr().is_null()
}

/// Thread-local callback that reports whether a raw address is owned
/// by the active backend's GC heap. Used by host-side allocators
/// (`pyre-object`'s `dealloc_items_block`) to discriminate
/// `try_gc_alloc_stable`-allocated blocks from `std::alloc`-backed
/// fallback blocks during the L1/L2 stepping-stone window:
/// `dealloc` must early-return for GC-managed pointers (the GC
/// sweeps them) and fall through to `std::alloc::dealloc` for
/// `std::alloc`-allocated ones.
pub type GcOwnsObjectFn = fn(addr: usize) -> bool;

thread_local! {
    static ACTIVE_GC_OWNS_OBJECT: Cell<Option<GcOwnsObjectFn>> = const { Cell::new(None) };
}

/// Install the active backend's `is_managed_heap_object` trampoline.
pub fn set_active_gc_owns_object(hook: Option<GcOwnsObjectFn>) {
    ACTIVE_GC_OWNS_OBJECT.with(|c| c.set(hook));
}

/// minimark.py:1900-1915 `id_or_identityhash` TLS hook.
pub type GcIdOrIdentityHashFn = fn(addr: usize) -> usize;

thread_local! {
    static ACTIVE_GC_ID_OR_IDENTITYHASH: Cell<Option<GcIdOrIdentityHashFn>> = const { Cell::new(None) };
}

pub fn set_active_gc_id_or_identityhash(hook: Option<GcIdOrIdentityHashFn>) {
    ACTIVE_GC_ID_OR_IDENTITYHASH.with(|c| c.set(hook));
}

/// Return a GC-move-stable address for identity hashing.
/// Falls back to `addr` when no backend is installed.
pub fn gc_id_or_identityhash(addr: usize) -> usize {
    ACTIVE_GC_ID_OR_IDENTITYHASH.with(|c| match c.get() {
        Some(f) => f(addr),
        None => addr,
    })
}

/// Whether `addr` lies inside the active backend's managed GC heap.
/// Returns `false` when no backend is installed on this thread —
/// callers treat that as "no GC owns this pointer" and fall through
/// to their non-GC dealloc path.
pub fn gc_owns_object(addr: usize) -> bool {
    ACTIVE_GC_OWNS_OBJECT.with(|c| match c.get() {
        Some(f) => f(addr),
        None => false,
    })
}

/// Return the current address for a managed object without treating it as a
/// root. During a minor collection this follows an already-installed nursery
/// forwarding pointer; otherwise it returns `addr` unchanged.
pub fn gc_current_object_address(addr: usize) -> usize {
    if addr == 0 || !gc_owns_object(addr) {
        return addr;
    }
    let hdr = unsafe { header::header_of(addr) };
    if unsafe { (*hdr).is_forwarded() } {
        unsafe { header::GcHeader::forwarding_address(hdr) }
    } else {
        addr
    }
}

/// Thread-local callbacks for registering/removing a Rust-stack slot
/// as a GC root with the currently active backend. Used by host-side
/// allocators whose callers need to keep a
/// just-allocated nursery pointer alive across a subsequent
/// potentially-collecting allocation.
///
/// RPython accomplishes the same thing automatically via its GC
/// transform pass (shadowstack save/restore around safepoints). pyre
/// lacks that pass, so root registration is explicit at the call
/// site. This is a documented TODO.
pub type AddRootFn = unsafe fn(slot: *mut GcRef);
pub type RemoveRootFn = fn(slot: *mut GcRef);

thread_local! {
    static ACTIVE_ADD_ROOT: Cell<Option<AddRootFn>> = const { Cell::new(None) };
    static ACTIVE_REMOVE_ROOT: Cell<Option<RemoveRootFn>> = const { Cell::new(None) };
}

/// Install the active backend's root-register callbacks. Pass `None`
/// to clear.
pub fn set_active_root_hooks(add: Option<AddRootFn>, remove: Option<RemoveRootFn>) {
    ACTIVE_ADD_ROOT.with(|c| c.set(add));
    ACTIVE_REMOVE_ROOT.with(|c| c.set(remove));
}

/// Register a stack slot as a GC root with the active backend. No-op
/// when no backend is installed on this thread.
///
/// # Safety
/// The caller must ensure the slot remains valid until
/// [`gc_remove_root`] is called with the same pointer.
pub unsafe fn gc_add_root(slot: *mut GcRef) {
    ACTIVE_ADD_ROOT.with(|c| {
        if let Some(f) = c.get() {
            unsafe { f(slot) }
        }
    });
}

/// Remove a previously-registered root slot from the active backend.
/// No-op when no backend is installed on this thread.
pub fn gc_remove_root(slot: *mut GcRef) {
    ACTIVE_REMOVE_ROOT.with(|c| {
        if let Some(f) = c.get() {
            f(slot)
        }
    });
}

/// Thread-local callback that performs a host-side write barrier through
/// the currently active backend GC.
pub type WriteBarrierFn = fn(obj: GcRef);

thread_local! {
    static ACTIVE_WRITE_BARRIER: Cell<Option<WriteBarrierFn>> = const { Cell::new(None) };
}

/// Install the active backend's write-barrier callback. Pass `None` to clear.
pub fn set_active_write_barrier(hook: Option<WriteBarrierFn>) {
    ACTIVE_WRITE_BARRIER.with(|c| c.set(hook));
}

/// Perform a write barrier through the active backend.
///
/// Calling convention: callers must invoke this before storing a GC reference
/// into `obj`, matching [`GcAllocator::write_barrier`]. The active callback is
/// thread-local and installed with [`set_active_write_barrier`] as a
/// [`WriteBarrierFn`]; this is a no-op when no barrier is installed on the
/// current thread.
pub fn gc_write_barrier(obj: GcRef) {
    ACTIVE_WRITE_BARRIER.with(|c| {
        if let Some(f) = c.get() {
            f(obj)
        }
    });
}
