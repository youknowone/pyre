use std::alloc::{Layout, alloc, alloc_zeroed, dealloc};
use std::ops::{Index, IndexMut};

use crate::{PY_NULL, PyObjectRef};

// The `GcArray(Float)` / `GcArray(Signed)` body and its no-collect allocation
// live with `rbigint`, whose `_digits` is lowered to one
// (`majit_rlib::lltypesystem::rlist`). The stable-tier constructors below are
// this crate's, because they root through this crate's shadow stack.
pub use majit_rlib::lltypesystem::rlist::{
    GC_FLOAT_ARRAY_GC_TYPE_ID, GC_INT_ARRAY_GC_TYPE_ID, TYPED_ITEMS_BLOCK_ITEMS_OFFSET,
    TYPED_ITEMS_BLOCK_LEN_OFFSET, TypedItemsBlock, alloc_typed_items_block_immortal,
    alloc_typed_items_block_nursery, itemsblock_gc_enabled, try_alloc_typed_items_block_nursery,
    try_typed_items_block_layout, typed_items_block_capacity, typed_items_block_clear,
    typed_items_block_items_base, typed_items_block_layout,
};

/// GC type id for the variable-length backing block of
/// `W_ListObject.items` / `W_TupleObject.wrappeditems` /
/// `DictStorage.values`. Shape matches RPython's
/// `GcArray(OBJECTPTR)` from `rpython/rtyper/lltypesystem/rlist.py:84,116`
/// — a `T_IS_VARSIZE` block with a single-slot `capacity` header followed by
/// inline `PyObjectRef` items. Registered from [`ITEMS_BLOCK_TOKEN`] with
/// `items_have_gc_ptrs=true` so the GC walks each item slot as a Ref.
/// Re-exported from `pyre_jit_trace::descr` for existing call sites.
pub const PY_OBJECT_ARRAY_GC_TYPE_ID: u32 = 9;

// 41 is `Ptr(GcArray(Signed))` and 42 is `Ptr(GcArray(Float))`, re-exported
// above: registered after the dictview (39) and getset-property (40) slots, so
// `gc.register_type` returns those two here.

/// GC type id for the `W_ObjectObject.storage` block — the mapdict instance
/// attribute-value array (`mapdict.py:910` `self.storage`, a
/// `Ptr(GcArray(OBJECTPTR))`). Distinct from `PY_OBJECT_ARRAY_GC_TYPE_ID` (9,
/// list/tuple items) so the block keeps its own identity even though every slot
/// is now a reference: a boxed attribute's value, or an
/// `UnboxedPlainAttribute`'s longlong list as a varsize leaf GcArray. The tid is
/// registered as a GC **leaf** (`items_have_gc_ptrs=false`) — the collector does
/// not walk the block's interior; the owning instance's
/// `object_object_custom_trace` walks it (`instance_walk_boxed_storage`).
/// Registered at the tail of the tid chain (after `W_COMPLEX_GC_TYPE_ID = 54`)
/// so no hardcoded constant shifts.
pub const W_MAPDICT_STORAGE_GC_TYPE_ID: u32 = 55;

/// The `(base_size, item_size, len_offset)` triple describing one varsize
/// GcArray block, derived from the Rust struct that carries the block's GC type
/// id — `symbolic.py:29 get_array_token`.
///
/// Upstream computes all three numbers from a single array lltype and hands
/// that one triple to every consumer: the JIT's array descriptor
/// (`descr.py:354 get_array_descr`), its length descriptor
/// (`descr.py:262 get_field_arraylen_descr`) and the collector's varsize shape
/// (`gctypelayout.py:273-291 encode_type_shape`, where `ofstovar`,
/// `varitemsize` and `ofstolength` all read the same `ARRAY`). Picking the
/// numbers apart per call site is what let the two scalar-array tids be
/// registered with `GcTypedArray`'s bare length word while the blocks actually
/// allocated under them were [`TypedItemsBlock`]s: the two structs agree on
/// 64-bit and differ by four bytes on wasm32, where the `[u64; 0]` items pad a
/// 4-byte length prefix out to 8. Every minor-collection copy of an
/// `rbigint._digits` block was then sized four bytes short and dropped the high
/// half of its last digit.
///
/// One token per (struct, element type) pair, so a consumer takes all three
/// numbers from the struct it means, or none of them.
pub struct ArrayToken {
    /// Offset of items[0] — the collector's `ofstovar`, the array descriptor's
    /// `base_size`.
    pub base_size: usize,
    /// `sizeof(ARRAY.OF)` — the element stride.
    pub item_size: usize,
    /// Offset of the length header — the collector's `ofstolength`.
    pub len_offset: usize,
}

/// `#[repr(C)] { capacity, items: [PyObjectRef; 0] }` — the single-block
/// inline-varsize GcArray body used by `W_ListObject.items` /
/// `W_TupleObject.items` / `DictStorage.values`.
/// Shape matches RPython's `GcArray(OBJECTPTR)` from
/// `rpython/rtyper/lltypesystem/rlist.py:84`: a length header at
/// offset 0 followed by inline items. Upstream's GcArray length IS
/// the allocated capacity (rlist.py:251 `len(l.items)` = allocated
/// slot count, fixed for the block's lifetime); live list length
/// lives on the enclosing `W_ListObject` wrapper per rlist.py:116
/// `("length", Signed)`.
///
/// Layout: offset 0 = `capacity` (= GcArray length header),
/// offset 8 = items[0..capacity]. Total allocation size =
/// `ITEMS_BLOCK_ITEMS_OFFSET + capacity * sizeof(PyObjectRef)`.
///
/// The header layout matches upstream and, by default, the allocator
/// now routes object-strategy blocks through the moving nursery
/// (`alloc_list_items_block_gc` / `alloc_tuple_items_block_gc` /
/// `grow_list_items_block_gc` → `try_gc_alloc(PY_OBJECT_ARRAY_GC_TYPE_ID,
/// cap)`), so `PY_OBJECT_ARRAY_GC_TYPE_ID` and the list/tuple custom
/// traces that forward the block-pointer field (eval.rs
/// `list_object_custom_trace` / `tuple_object_custom_trace`) are live at
/// collection time. `W_ListObject` / `W_TupleObject` hold
/// `{length: usize, items: *mut ItemsBlock}` fields directly (no
/// `PyObjectArray` fat wrapper for list/tuple). The `std::alloc`
/// `alloc_items_block` / `grow_items_block` below are the
/// `PYRE_GC_ITEMSBLOCK=0` fallback (provably identical pre-migration
/// behaviour).
#[repr(C)]
pub struct ItemsBlock {
    /// Allocated capacity — treated as the GcArray-length header
    /// (rlist.py:251 `len(l.items)`). The GC registration sets
    /// `length_offset=0` to this field so the walker iterates
    /// `0..capacity`. Fixed from `alloc_items_block()` through
    /// `dealloc_items_block()`; a `grow_items_block()` call
    /// allocates a fresh block rather than mutating this field.
    pub capacity: usize,
    /// Items inline after the header. Size known only at allocation
    /// time — accessed via pointer arithmetic from
    /// `items_block_items_base()`.
    items: [PyObjectRef; 0],
}

pub const ITEMS_BLOCK_ITEMS_OFFSET: usize = std::mem::offset_of!(ItemsBlock, items);

/// Offset of the `capacity` header the collector reads as the GcArray length.
pub const ITEMS_BLOCK_LEN_OFFSET: usize = std::mem::offset_of!(ItemsBlock, capacity);

/// `get_array_token(GcArray(OBJECTPTR))` — the shape [`PY_OBJECT_ARRAY_GC_TYPE_ID`]
/// is registered with and `pyobject_gcarray_descr` carries.
pub const ITEMS_BLOCK_TOKEN: ArrayToken = ArrayToken {
    base_size: ITEMS_BLOCK_ITEMS_OFFSET,
    item_size: std::mem::size_of::<PyObjectRef>(),
    len_offset: ITEMS_BLOCK_LEN_OFFSET,
};

/// Return the items base pointer (i.e. `&items[0]`) of an
/// `ItemsBlock`. Null-safe: returns a null `*mut PyObjectRef` if the
/// block itself is null, so callers can treat a null items pointer as
/// an empty list without branching through `Option`.
#[inline]
pub unsafe fn items_block_items_base(block: *mut ItemsBlock) -> *mut PyObjectRef {
    if block.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { (block as *mut u8).add(ITEMS_BLOCK_ITEMS_OFFSET) as *mut PyObjectRef }
}

/// Allocated capacity (GcArray length header) of an `ItemsBlock`.
/// Returns 0 for a null pointer so "empty list" is represented by
/// a null `items` field.
#[inline]
pub unsafe fn items_block_capacity(block: *mut ItemsBlock) -> usize {
    if block.is_null() {
        return 0;
    }
    unsafe { (*block).capacity }
}

/// Allocate a fresh `ItemsBlock` populated with the given values. The
/// capacity is `values.len().max(1)`; unused slots past `values.len()`
/// are NULL-initialised so the GC walker (once Phase L2 activates
/// `PY_OBJECT_ARRAY_GC_TYPE_ID`) sees valid NULL refs past the live
/// prefix — upstream `gc_malloc_array` zero-fills (rlist.py:262-267
/// `_ll_list_resize_really`). Used by `W_ListObject::from_vec`.
///
/// The `max(1)` clamp is the list-strategy overallocation policy
/// (rlist.py:251 `_ll_list_resize_*` always keeps at least one slot
/// for in-place growth). Tuples must NOT use this allocator — see
/// [`alloc_tuple_items_block`] for the exact-size variant.
pub unsafe fn alloc_list_items_block(values: &[PyObjectRef]) -> *mut ItemsBlock {
    let len = values.len();
    let cap = len.max(1);
    unsafe {
        let block = alloc_items_block(cap);
        let base = items_block_items_base(block);
        for (i, v) in values.iter().enumerate() {
            *base.add(i) = *v;
        }
        for i in len..cap {
            *base.add(i) = PY_NULL;
        }
        block
    }
}

/// `pypy/objspace/std/tupleobject.py:376-390` `W_TupleObject`
/// allocator. Allocates an `ItemsBlock` with capacity exactly equal to
/// `values.len()` — tuples are immutable so the GcArray header
/// `length` IS the live tuple length (no overallocation room). For an
/// empty tuple this yields a 0-cap header-only block; the GcArray
/// pointer is non-null but addresses zero items.
///
/// Read length back via `arraylen_gc(items_block, pyobject_gcarray_descr)`
/// or [`items_block_capacity`] on the host side. No companion length
/// cache lives on `W_TupleObject` (`_immutable_fields_ =
/// ['wrappeditems[*]']` per upstream tupleobject.py:381).
pub unsafe fn alloc_tuple_items_block(values: &[PyObjectRef]) -> *mut ItemsBlock {
    let cap = values.len();
    unsafe {
        let block = alloc_items_block(cap);
        let base = items_block_items_base(block);
        for (i, v) in values.iter().enumerate() {
            *base.add(i) = *v;
        }
        block
    }
}

/// Grow an `ItemsBlock` to `new_cap` capacity, copying `live_len`
/// existing items from `old` and NULL-initialising the rest. Returns the new
/// block. The caller swaps the owner field and then deallocates `old`, keeping
/// the SETFIELD_GC barrier adjacent to the publication. `old` may be null
/// (fresh allocation). rlist.py:262-267 parity.
pub unsafe fn grow_list_items_block(
    old: *mut ItemsBlock,
    new_cap: usize,
    live_len: usize,
) -> *mut ItemsBlock {
    unsafe { grow_items_block(old, new_cap, live_len) }
}

/// Deallocate an `ItemsBlock` previously allocated via
/// `alloc_list_items_block` / `grow_list_items_block`. No-op on null.
pub unsafe fn dealloc_list_items_block(block: *mut ItemsBlock) {
    unsafe { dealloc_items_block(block) }
}

// ─── mapdict instance-storage block: stable GcArray(OBJECTPTR) ────────────
//
// `W_ObjectObject.storage` (`mapdict.py:910` `self.storage`) is a
// `Ptr(GcArray(OBJECTPTR))`. Unlike list/tuple item blocks (nursery,
// `PY_OBJECT_ARRAY_GC_TYPE_ID`, inline-traced), the mapdict storage is a MIXED
// boxed/unboxed array, so it is allocated `stable` (non-moving old-gen) under a
// LEAF tid (`W_MAPDICT_STORAGE_GC_TYPE_ID`) and its interior is walked only by
// the instance's `object_object_custom_trace` (map-masked). Stable allocation
// mirrors the instance's own `try_gc_alloc_stable` (objectobject.rs) and the
// `TypedItemsBlock` int/float backing blocks: a non-moving block means the
// instance's `storage` pointer never needs rewriting on a minor GC, and the
// A stable allocation does not itself start a collection, but it is still a GC
// operation and can wait behind a collection started by another mutator.
// Therefore its inputs and fresh result need the same shadow-stack publication
// as nursery allocation.

/// Allocate a fresh stable `ItemsBlock` holding `values` in its first slots and
/// NULL in the rest, tagged `W_MAPDICT_STORAGE_GC_TYPE_ID` (leaf). The map is
/// the length authority (mapdict.py:942-959), so `cap` is an allocation bound
/// rather than a length; a live instance passes the larger of its current
/// capacity and `values.len()` so the capacity never shrinks. Every slot is
/// written, so the block is safe to expose to the collector immediately. Falls
/// back to the `std::alloc` [`alloc_items_block`] when no GC hook is installed
/// (pure interpreter / early startup). `cap` 0 yields a header-only block.
pub unsafe fn alloc_instance_items_block(values: &[PyObjectRef], cap: usize) -> *mut ItemsBlock {
    debug_assert!(cap >= values.len());
    let _roots = crate::gc_roots::push_roots();
    let values_base = crate::gc_roots::pin_roots(values);
    unsafe {
        let block = alloc_mapdict_storage_block(cap);
        crate::gc_roots::pin_root(block as PyObjectRef);
        let block =
            crate::gc_roots::shadow_stack_get(values_base + values.len()) as *mut ItemsBlock;
        let base = items_block_items_base(block);
        for i in 0..values.len() {
            *base.add(i) = crate::gc_roots::shadow_stack_get(values_base + i);
        }
        for i in values.len()..cap {
            *base.add(i) = PY_NULL;
        }
        block
    }
}

/// Grow (or shrink) an instance storage block to `new_cap`, copying `live_len`
/// existing items from `old` and NULL-filling any spare tail. The caller keeps
/// the owning instance rooted, installs and barriers the new block, then
/// deallocates `old`; dropping it here would introduce a GC operation while the
/// fresh block is not yet reachable from its owner. `old` may be null.
/// mapdict.py:942-959 `_add_attr` grow-by-one.
pub unsafe fn grow_instance_items_block(
    old: *mut ItemsBlock,
    new_cap: usize,
    live_len: usize,
) -> *mut ItemsBlock {
    unsafe {
        let fresh = alloc_mapdict_storage_block(new_cap);
        let new_base = items_block_items_base(fresh);
        let copy = live_len.min(new_cap);
        if !old.is_null() && copy > 0 {
            std::ptr::copy_nonoverlapping(items_block_items_base(old), new_base, copy);
        }
        for i in copy..new_cap {
            *new_base.add(i) = PY_NULL;
        }
        fresh
    }
}

/// Deallocate an instance storage block. A GC-managed (stable) block is
/// reclaimed by the collector and must never be freed here; the `std::alloc`
/// fallback block is freed. No-op on null. `try_gc_owns_object` discriminates.
pub unsafe fn dealloc_instance_items_block(block: *mut ItemsBlock) {
    unsafe { dealloc_items_block(block) }
}

/// Stable leaf-block allocator for mapdict storage. Routes through
/// `try_gc_alloc_stable(W_MAPDICT_STORAGE_GC_TYPE_ID, payload)`; the capacity
/// header is set, items are left uninitialised (the caller writes every slot
/// before exposing the block). Falls back to `std::alloc` [`alloc_items_block`]
/// when no GC hook is installed. `cap` may be zero (header-only block).
unsafe fn alloc_mapdict_storage_block(cap: usize) -> *mut ItemsBlock {
    let payload = ITEMS_BLOCK_ITEMS_OFFSET + cap * std::mem::size_of::<PyObjectRef>();
    let raw = crate::gc_hook::try_gc_alloc_stable_raw(W_MAPDICT_STORAGE_GC_TYPE_ID, payload);
    if !raw.is_null() {
        let block = raw as *mut ItemsBlock;
        unsafe { (*block).capacity = cap };
        return block;
    }
    unsafe { alloc_items_block(cap) }
}

/// Phase L2 nursery allocation of a fresh `ItemsBlock` with `cap` slots,
/// as a `PY_OBJECT_ARRAY_GC_TYPE_ID` varsize GcArray. The collector reads
/// its length from the offset-0 `capacity` header and forwards
/// items[0..capacity], so the caller MUST write every slot (value or
/// NULL) before a collection can observe the block. Falls back to the
/// `std::alloc` [`alloc_items_block`] when the gate is off or no GC hook
/// is installed (pure interpreter / early startup). Items are left
/// uninitialised either way.
unsafe fn alloc_items_block_gc(cap: usize) -> *mut ItemsBlock {
    if itemsblock_gc_enabled() {
        let payload = ITEMS_BLOCK_ITEMS_OFFSET + cap * std::mem::size_of::<PyObjectRef>();
        if let Some(raw) = crate::gc_hook::try_gc_alloc(PY_OBJECT_ARRAY_GC_TYPE_ID, payload) {
            if !raw.is_null() {
                let block = raw as *mut ItemsBlock;
                unsafe { (*block).capacity = cap };
                return block;
            }
        }
    }
    unsafe { alloc_items_block(cap) }
}

/// List-construction allocator on the Phase L2 nursery path. Pins each
/// element across the (collecting) block allocation and fills from the
/// relocated shadow-stack slots — the `pop_roots` read-back of
/// `w_tuple_new_array_backed` — because the local `values` slice still
/// holds pre-collection addresses. Capacity is `len.max(1)`
/// (overallocation policy); spare slots are NULL. Degrades to the
/// `std::alloc` [`alloc_list_items_block`] when the gate is off.
pub unsafe fn alloc_list_items_block_gc(values: &[PyObjectRef]) -> *mut ItemsBlock {
    if !itemsblock_gc_enabled() {
        return unsafe { alloc_list_items_block(values) };
    }
    let len = values.len();
    let cap = len.max(1);
    let _roots = crate::gc_roots::push_roots();
    let save = crate::gc_roots::shadow_stack_len();
    for &v in values {
        crate::gc_roots::pin_root(v);
    }
    let block_slot = crate::gc_roots::shadow_stack_len();
    let block = unsafe { alloc_items_block_gc(cap) };
    // `alloc_items_block_gc` returns a fresh GCREF.  RPython's
    // gct_fv_gc_malloc pop_roots makes that result live before the next
    // collecting operation.  Publish it before the ownership query / write
    // barrier below: either operation can park behind another thread's
    // collection, which may move the otherwise-unrooted fresh array.
    crate::gc_roots::pin_root(block as PyObjectRef);
    let block = crate::gc_roots::shadow_stack_get(block_slot) as *mut ItemsBlock;
    let base = unsafe { items_block_items_base(block) };
    if len > 0 {
        // RPython's pop_roots reloads the livevars as one generated block.
        // Enter TLS once for the whole contiguous item range instead of once
        // per element; `base` names the freshly allocated block's initialized
        // prefix and cannot alias the shadow stack.
        let dst = unsafe { std::slice::from_raw_parts_mut(base, len) };
        crate::gc_roots::shadow_stack_copy_range(save, dst);
    }
    for i in len..cap {
        unsafe { *base.add(i) = PY_NULL };
    }
    // Old→young barrier if the block landed in old-gen (see
    // alloc_tuple_items_block_gc): registers an old-gen block holding young
    // elements onto the remembered set, no-op for a nursery block.
    if crate::gc_hook::try_gc_owns_object(block as *mut u8) {
        crate::gc_hook::try_gc_write_barrier(block as *mut u8);
    }
    block
}

/// List-grow allocator on the Phase L2 nursery path. Pins the OLD BLOCK across
/// the new (collecting) allocation, copies its `live_len` items into the new
/// one and NULL-fills the spare tail. The caller installs the new owner edge
/// through its write barrier and then hands the old block to
/// [`dealloc_list_items_block`] (which no-ops on a GC-managed block — the
/// collector reclaims it). `old` may be null with `live_len == 0` (fresh
/// allocation). Degrades to the `std::alloc` [`grow_list_items_block`] when the
/// gate is off.
///
/// The block itself is the root, not its items. A block is a traced GC array of
/// GC pointers (`PY_OBJECT_ARRAY_GC_TYPE_ID` is registered `TypeInfo::varsize`
/// over `PyObjectRef` slots, `pyre-jit/src/eval.rs`), so a collection during the
/// allocation below keeps every item reachable through it and relocates the
/// slots in place — reading them back afterwards yields post-collection
/// addresses without naming them one at a time. That is also the upstream
/// shape: `_ll_list_resize_really` (rlist.py:262-267) mallocs the new array and
/// `ll_arraycopy`s into it, with no per-item root bracket anywhere.
///
/// Pinning per item instead made a grow cost `2 * live_len` GC-ownership
/// queries — each a TLS lookup, a `RefCell` borrow, an arena range test and a
/// rawmalloced-set probe — so the 1% of appends that resize dominated the
/// profile of every Object-strategy list build.
pub unsafe fn grow_list_items_block_gc(
    old: *mut ItemsBlock,
    new_cap: usize,
    live_len: usize,
) -> *mut ItemsBlock {
    if !itemsblock_gc_enabled() {
        return unsafe { grow_list_items_block(old, new_cap, live_len) };
    }
    let _roots = crate::gc_roots::push_roots();
    let save = crate::gc_roots::shadow_stack_len();
    // Only a GC-owned block needs rooting: `alloc_items_block_gc` falls back to
    // `std::alloc` when the nursery declines, and such a block never moves, so
    // the incoming pointer stays valid on its own. A null `old` carries no items.
    let root_old = !old.is_null() && crate::gc_hook::try_gc_owns_object(old as *mut u8);
    if root_old {
        crate::gc_roots::pin_root(old as crate::pyobject::PyObjectRef);
    }
    let new_block_slot = crate::gc_roots::shadow_stack_len();
    let new_block = unsafe { alloc_items_block_gc(new_cap) };
    crate::gc_roots::pin_root(new_block as PyObjectRef);
    let new_block = crate::gc_roots::shadow_stack_get(new_block_slot) as *mut ItemsBlock;
    let new_base = unsafe { items_block_items_base(new_block) };
    let old = if root_old {
        crate::gc_roots::shadow_stack_get(save) as *mut ItemsBlock
    } else {
        old
    };
    if live_len > 0 {
        let old_base = unsafe { items_block_items_base(old) };
        unsafe { std::ptr::copy_nonoverlapping(old_base, new_base, live_len) };
    }
    for i in live_len..new_cap {
        unsafe { *new_base.add(i) = PY_NULL };
    }
    // Old→young barrier if the grown block landed in old-gen (see
    // alloc_tuple_items_block_gc).
    if crate::gc_hook::try_gc_owns_object(new_block as *mut u8) {
        crate::gc_hook::try_gc_write_barrier(new_block as *mut u8);
    }
    new_block
}

/// Tuple-construction allocator on the Phase L2 nursery path. Exact-size
/// (`cap == len` — tuples are immutable and every slot is written, no
/// overallocation). Pins each element across the (collecting) block
/// allocation and fills from the relocated shadow-stack slots, mirroring
/// `w_tuple_new_array_backed`'s read-back. Degrades to the `std::alloc`
/// [`alloc_tuple_items_block`] when the gate is off.
pub unsafe fn alloc_tuple_items_block_gc(values: &[PyObjectRef]) -> *mut ItemsBlock {
    if !itemsblock_gc_enabled() {
        return unsafe { alloc_tuple_items_block(values) };
    }
    let cap = values.len();
    let _roots = crate::gc_roots::push_roots();
    let save = crate::gc_roots::shadow_stack_len();
    for &v in values {
        crate::gc_roots::pin_root(v);
    }
    let block_slot = crate::gc_roots::shadow_stack_len();
    let block = unsafe { alloc_items_block_gc(cap) };
    crate::gc_roots::pin_root(block as PyObjectRef);
    let block = crate::gc_roots::shadow_stack_get(block_slot) as *mut ItemsBlock;
    let base = unsafe { items_block_items_base(block) };
    if cap > 0 {
        // Same block-shaped pop_roots reload as the list constructor above.
        let dst = unsafe { std::slice::from_raw_parts_mut(base, cap) };
        crate::gc_roots::shadow_stack_copy_range(save, dst);
    }
    // The block may have landed in old-gen (nursery-full fallback) while its
    // elements are still young. That old→young edge is invisible to a minor
    // collection unless the block is on the remembered set, so write-barrier
    // it here. A nursery block carries no TRACK_YOUNG_PTRS and the barrier is
    // a no-op; an old-gen block is registered so the next minor collection
    // walks its items (write_barrier_from_array, incminimark.py:1495). Guard
    // on GC ownership exactly like `list_write_barrier`.
    if crate::gc_hook::try_gc_owns_object(block as *mut u8) {
        crate::gc_hook::try_gc_write_barrier(block as *mut u8);
    }
    block
}

/// Allocate an exact-`cap` NULL-filled GC-managed `ItemsBlock` of refs for
/// the walker's recording-time materialization of a virtual
/// `NEW_ARRAY_CLEAR` (BUILD_LIST / BUILD_TUPLE). Returns `None` when the gate
/// is off or no GC hook is installed, so the caller declines materialization
/// rather than handing back a `std::alloc` block that would never be freed.
/// The block is a `PY_OBJECT_ARRAY_GC_TYPE_ID` varsize array, so its items are
/// GC-traced and the NULL prefix is benign until the caller fills slots via
/// `setarrayitem_ref` + a block write barrier. No `std::alloc` fallback here
/// (unlike [`alloc_items_block_gc`]): the materialization requires a GC-traced
/// block whose escaping young element refs are forwarded by a collection.
pub unsafe fn alloc_cleared_ref_items_block_gc(cap: usize) -> Option<*mut ItemsBlock> {
    if !itemsblock_gc_enabled() {
        return None;
    }
    let payload = ITEMS_BLOCK_ITEMS_OFFSET + cap * std::mem::size_of::<PyObjectRef>();
    let raw = crate::gc_hook::try_gc_alloc(PY_OBJECT_ARRAY_GC_TYPE_ID, payload)?;
    if raw.is_null() {
        return None;
    }
    let block = raw as *mut ItemsBlock;
    unsafe {
        (*block).capacity = cap;
        let base = items_block_items_base(block);
        for i in 0..cap {
            *base.add(i) = PY_NULL;
        }
    }
    Some(block)
}

/// Allocate a fresh `ItemsBlock` with the given capacity via
/// `std::alloc::alloc`. This is the `PYRE_GC_ITEMSBLOCK=0` fallback;
/// the default path is the moving-nursery `alloc_items_block_gc` above
/// (`try_gc_alloc`, caller-side root tracking via `gc_roots::pin_root` +
/// the block write-barrier). An earlier `try_gc_alloc_stable` attempt was
/// abandoned because old-gen (non-moving) allocation accumulates
/// per-iteration containers until a major GC; the nursery path avoids that
/// (minor GC reclaims short-lived blocks) and is perf-neutral on
/// fannkuch/nbody.
///
/// The capacity header is initialized; items are left uninitialized —
/// the caller must write all `capacity` slots before exposing the
/// pointer to the GC walker. `cap` may be zero; the resulting block
/// holds only the 8-byte capacity header (used by tuple — see
/// [`alloc_tuple_items_block`] — for empty tuples).
unsafe fn alloc_items_block(cap: usize) -> *mut ItemsBlock {
    let layout = items_block_layout(cap);
    unsafe {
        let raw = alloc(layout);
        if raw.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let block = raw as *mut ItemsBlock;
        (*block).capacity = cap;
        block
    }
}

/// Deallocate an `ItemsBlock` previously allocated via
/// [`alloc_items_block`] or [`grow_items_block`]. Phase L2: a
/// GC-managed block (nursery / old-gen) is reclaimed by the collector
/// and must never be freed here — its allocation is prefixed by a
/// `GcHeader` the `std::alloc` layout knows nothing about, so handing
/// it to `dealloc` would corrupt the allocator. `try_gc_owns_object`
/// discriminates the two block origins during cutover.
unsafe fn dealloc_items_block(block: *mut ItemsBlock) {
    if block.is_null() {
        return;
    }
    if crate::gc_hook::try_gc_owns_object(block as *mut u8) {
        return;
    }
    unsafe {
        let cap = (*block).capacity;
        let layout = items_block_layout(cap);
        dealloc(block as *mut u8, layout);
    }
}

fn items_block_layout(cap: usize) -> Layout {
    let total = ITEMS_BLOCK_ITEMS_OFFSET + cap * std::mem::size_of::<PyObjectRef>();
    Layout::from_size_align(total, std::mem::align_of::<ItemsBlock>()).expect("ItemsBlock layout")
}

/// Return the items base pointer of an `ItemsBlock`.
#[inline]
unsafe fn items_block_items_ptr(block: *mut ItemsBlock) -> *mut PyObjectRef {
    unsafe { (block as *mut u8).add(ITEMS_BLOCK_ITEMS_OFFSET) as *mut PyObjectRef }
}

/// Reallocate an `ItemsBlock` to a new capacity, copying live items.
/// Spare slots `live_len..capacity` are NULL-initialized so the GC
/// walker (once `PY_OBJECT_ARRAY_GC_TYPE_ID` is active on the
/// allocation) sees valid NULL refs in unused slots — upstream
/// relies on `gc_malloc_array` zero-filling the fresh block
/// (rlist.py:262-267 `_ll_list_resize_really`); pyre's `alloc`
/// uses `std::alloc::alloc` which is not zero-filled so we
/// explicit-init here.
/// Old block is deallocated. Returns the new block.
unsafe fn grow_items_block(
    old: *mut ItemsBlock,
    new_cap: usize,
    live_len: usize,
) -> *mut ItemsBlock {
    unsafe {
        let fresh = alloc_items_block(new_cap);
        let new_base = items_block_items_ptr(fresh);
        if !old.is_null() && live_len > 0 {
            std::ptr::copy_nonoverlapping(items_block_items_ptr(old), new_base, live_len);
        }
        let fresh_cap = (*fresh).capacity;
        for i in live_len..fresh_cap {
            *new_base.add(i) = PY_NULL;
        }
        fresh
    }
}

// ─── TypedItemsBlock: GcArray(Float)/GcArray(Signed) backing block ───────
//
// The block itself is `majit_rlib::lltypesystem::rlist`, re-exported at the top
// of this file: the Float / Integer list strategies (`listobject.py`
// FloatListStrategy / IntegerListStrategy) and `rbigint._digits` are the same
// `erase([float])` / `erase([int])` lowering, so one body serves all three.
//
// What remains here is the stable (old-gen) tier the list strategies allocate
// in, which roots through this crate's shadow stack, plus the array tokens the
// descr registry reads.

/// `get_array_token(GcArray(Signed))` — [`GC_INT_ARRAY_GC_TYPE_ID`].
pub const TYPED_ITEMS_BLOCK_INT_TOKEN: ArrayToken = ArrayToken {
    base_size: TYPED_ITEMS_BLOCK_ITEMS_OFFSET,
    item_size: std::mem::size_of::<i64>(),
    len_offset: TYPED_ITEMS_BLOCK_LEN_OFFSET,
};

/// `get_array_token(GcArray(Float))` — [`GC_FLOAT_ARRAY_GC_TYPE_ID`]. A distinct
/// ARRAY identity from the signed one, sharing this block's layout.
pub const TYPED_ITEMS_BLOCK_FLOAT_TOKEN: ArrayToken = ArrayToken {
    base_size: TYPED_ITEMS_BLOCK_ITEMS_OFFSET,
    item_size: std::mem::size_of::<f64>(),
    len_offset: TYPED_ITEMS_BLOCK_LEN_OFFSET,
};

/// The items are `u64` words, so an element type wider or narrower than the
/// storage word would stride past or short of them.
const _: () = assert!(
    TYPED_ITEMS_BLOCK_INT_TOKEN.item_size == std::mem::size_of::<u64>()
        && TYPED_ITEMS_BLOCK_FLOAT_TOKEN.item_size == std::mem::size_of::<u64>(),
    "TypedItemsBlock stores u64 words; another element width needs its own block",
);

/// Allocate a fresh zero-filled `TypedItemsBlock` with the given capacity, as a
/// `tid` (`GC_INT_ARRAY` / `GC_FLOAT_ARRAY`) varsize GcArray. Zero-fill matches
/// `gc_malloc_array` (rlist.py:262-267 `_ll_list_resize_really`) and the
/// Float/Int strategy `_none_value` (0.0 / 0); `try_gc_alloc_stable` memory is
/// not guaranteed zeroed, so the items are cleared explicitly. `cap` is clamped
/// to at least 1 (rlist.py:251 overallocation policy). Falls back to
/// `std::alloc::alloc_zeroed` when the gate is off or no GC hook is installed
/// (pure interpreter / early startup).
///
/// The block is allocated `stable` (old-gen, mark-sweep, non-moving) — the same
/// tier as its `W_ListObject` owner (`listobject.rs` `try_gc_alloc_stable`).
/// The items are plain scalars (no GC pointers), so the block is a varsize leaf
/// the collector marks: it never relocates and never holds a young pointer, so
/// the owning list needs no remembered-set barrier when the block changes. Its
/// `int_items.block` / `float_items.block` owner slot is marked-live by
/// `list_object_custom_trace` — but only once that slot holds the block. Being
/// old-gen buys immobility, not survival: a block born before this cycle
/// finished scanning carries no `GCFLAG_VISITED`, so between this call and the
/// owner store the caller must root it (`IntArray::pin_block`).
pub unsafe fn alloc_typed_items_block(cap: usize, tid: u32) -> *mut TypedItemsBlock {
    unsafe {
        try_alloc_typed_items_block(cap, tid)
            .unwrap_or_else(|| std::alloc::handle_alloc_error(Layout::new::<TypedItemsBlock>()))
    }
}

/// Fallible `gc_malloc_array` surface used where the RPython caller exposes
/// allocation failure as `MemoryError` (notably `rbigint.lshift`).  The
/// ordinary translated allocation path above remains infallible at the Rust
/// type level, matching RPython's implicit exception edge; this entry point
/// makes that edge explicit for Rust methods that already return `Result`.
pub unsafe fn try_alloc_typed_items_block(cap: usize, tid: u32) -> Option<*mut TypedItemsBlock> {
    let cap = cap.max(1);
    let layout = try_typed_items_block_layout(cap)?;
    let payload = layout.size();
    if itemsblock_gc_enabled() {
        let raw = crate::gc_hook::try_gc_alloc_stable_raw(tid, payload);
        if !raw.is_null() {
            let block = raw as *mut TypedItemsBlock;
            unsafe {
                (*block).capacity = cap;
                std::ptr::write_bytes(
                    typed_items_block_items_base(block),
                    0,
                    cap * std::mem::size_of::<u64>(),
                );
            }
            return Some(block);
        }
    }
    unsafe {
        let raw = alloc_zeroed(layout);
        if raw.is_null() {
            return None;
        }
        let block = raw as *mut TypedItemsBlock;
        (*block).capacity = cap;
        Some(block)
    }
}

/// Grow a `TypedItemsBlock` to `new_cap`, copying `live_len` words from `old`,
/// zero-filling the rest, and deallocating `old`. `old` may be null.
/// rlist.py:262-267 parity. `old` is allocated `stable` (old-gen, non-moving),
/// so it keeps its address across the (possibly collecting) allocation of
/// `fresh` and the live words are copied directly; it also stays reachable
/// through its owner's `block` field for that whole span, since the caller only
/// overwrites that field with the returned value.
pub unsafe fn grow_typed_items_block(
    old: *mut TypedItemsBlock,
    new_cap: usize,
    live_len: usize,
    tid: u32,
) -> *mut TypedItemsBlock {
    unsafe {
        let fresh = alloc_typed_items_block(new_cap, tid);
        // `fresh` has no owner field yet, and `dealloc_typed_items_block` below
        // enters a GC operation (`try_gc_owns_object`) that parks this mutator
        // while another thread can run a cycle to `STATE_SWEEPING`. Old-gen is
        // mark-sweep, so an unreachable, not-yet-`VISITED` block is swept —
        // root it for the span, as `gct_fv_gc_malloc` roots every livevar it
        // brackets a malloc with (`framework.py:853-856`).
        let _roots = crate::gc_roots::push_roots();
        let fresh_root = crate::gc_roots::shadow_stack_len();
        crate::gc_roots::pin_root(fresh as crate::PyObjectRef);
        if !old.is_null() && live_len > 0 {
            std::ptr::copy_nonoverlapping(
                typed_items_block_items_base(old),
                typed_items_block_items_base(fresh),
                live_len * std::mem::size_of::<u64>(),
            );
        }
        if !old.is_null() {
            dealloc_typed_items_block(old);
        }
        crate::gc_roots::shadow_stack_get(fresh_root) as *mut TypedItemsBlock
    }
}

/// Deallocate a `TypedItemsBlock`. No-op on null. A GC-managed block is reclaimed
/// by the collector and must never be freed here — its allocation is prefixed by
/// a `GcHeader` the `std::alloc` layout knows nothing about. `try_gc_owns_object`
/// gates the `std::alloc` free to the gate-off / no-hook fallback blocks.
pub unsafe fn dealloc_typed_items_block(block: *mut TypedItemsBlock) {
    if block.is_null() {
        return;
    }
    if crate::gc_hook::try_gc_owns_object(block as *mut u8) {
        return;
    }
    unsafe {
        let cap = (*block).capacity;
        let layout = typed_items_block_layout(cap);
        dealloc(block as *mut u8, layout);
    }
}

// ─── FixedObjectArray: pyframe.py:112 make_sure_not_resized parity ────────
//
// RPython `locals_cells_stack_w = [None] * size; make_sure_not_resized(...)`
// becomes a fixed-length GcArray (`Ptr(GcArray(PyObjectRef))`). The layout
// here matches that upstream shape so `GETFIELD_GC_R(frame, locals_cells_stack)
// + GETARRAYITEM_GC_R(array_ptr, i)` means the same thing in both worlds:
// single-indirection, items immediately after the length header.
//
// Layout: `[len: usize] [items: PyObjectRef; len]` (variable-length,
// flexible-array tail). Allocation happens via a custom `Layout` at the
// caller site (see `pyre_interpreter::pyframe::alloc_fixed_array_with_header`).

/// Offset of the length prefix within `FixedObjectArray`.
pub const FIXED_ARRAY_LEN_OFFSET: usize = std::mem::offset_of!(FixedObjectArray, len);

/// Offset of the first item within `FixedObjectArray` (immediately after
/// the length prefix). The JIT-visible array descriptor uses this as
/// `base_size` so `GETARRAYITEM_GC_*` reads items directly.
pub const FIXED_ARRAY_ITEMS_OFFSET: usize = std::mem::offset_of!(FixedObjectArray, _items);

/// `get_array_token(GcArray(PyObjectRef))` for the frame-locals and mro blocks.
pub const FIXED_OBJECT_ARRAY_TOKEN: ArrayToken = ArrayToken {
    base_size: FIXED_ARRAY_ITEMS_OFFSET,
    item_size: std::mem::size_of::<PyObjectRef>(),
    len_offset: FIXED_ARRAY_LEN_OFFSET,
};

/// `FixedObjectArray` blocks are allocated under [`PY_OBJECT_ARRAY_GC_TYPE_ID`]
/// as well (`pyframe::alloc_frame_locals_array`, [`alloc_mro_block_gc`]), and one
/// tid carries one varsize shape — the one registered from [`ITEMS_BLOCK_TOKEN`].
/// Both structs are a length word followed by `[PyObjectRef; 0]` today, so the
/// two shapes coincide on every target. Diverge them and the collector would
/// size one of the two wrong, which on a 32-bit target is a short copy rather
/// than a build error. Keep it a build error.
const _: () = assert!(
    FIXED_OBJECT_ARRAY_TOKEN.base_size == ITEMS_BLOCK_TOKEN.base_size
        && FIXED_OBJECT_ARRAY_TOKEN.item_size == ITEMS_BLOCK_TOKEN.item_size
        && FIXED_OBJECT_ARRAY_TOKEN.len_offset == ITEMS_BLOCK_TOKEN.len_offset,
    "FixedObjectArray and ItemsBlock share one GcArray tid; their shapes must match",
);

/// pyframe.py:110-112: fixed-length GcArray for `locals_cells_stack_w`.
///
/// Once created, the length never changes. No push, no grow.
/// Items are mutable (stack operations write via index) but the
/// array cannot be resized.
///
/// `_items` is a zero-sized flexible-array marker: the real items live
/// immediately after `len` in the allocation, accessed via pointer
/// arithmetic through `items_ptr` / `as_slice`.
#[repr(C)]
pub struct FixedObjectArray {
    /// Length prefix. Matches RPython `Ptr(GcArray(T))` header so that
    /// the JIT's arraydescr `base_size = FIXED_ARRAY_ITEMS_OFFSET` lands
    /// on items[0].
    pub len: usize,
    /// Flexible-array tail marker. Actual items follow immediately in
    /// memory (sized to `len` at allocation time); this field has size 0.
    _items: [PyObjectRef; 0],
}

impl FixedObjectArray {
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn items_ptr(&self) -> *const PyObjectRef {
        unsafe {
            (self as *const Self as *const u8).add(FIXED_ARRAY_ITEMS_OFFSET) as *const PyObjectRef
        }
    }

    #[inline]
    pub fn items_mut_ptr(&mut self) -> *mut PyObjectRef {
        unsafe { (self as *mut Self as *mut u8).add(FIXED_ARRAY_ITEMS_OFFSET) as *mut PyObjectRef }
    }

    pub fn as_slice(&self) -> &[PyObjectRef] {
        unsafe { std::slice::from_raw_parts(self.items_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [PyObjectRef] {
        unsafe { std::slice::from_raw_parts_mut(self.items_mut_ptr(), self.len) }
    }

    /// Store a GC reference through the host interpreter's equivalent of the
    /// GC transform's `setarrayitem_gc` rewrite.
    ///
    /// RPython keeps the value live in the translated shadow stack, emits the
    /// conditional array write barrier, then performs the store.  A raw Rust
    /// local is not part of that generated root map, so publish it explicitly
    /// and reload it after the barrier's safepoint before writing the slot.
    #[inline]
    pub fn set_ref(&mut self, index: usize, value: PyObjectRef) {
        assert!(index < self.len);
        // Clearing a slot cannot create an old-to-young edge.  PyPy's
        // `popvalue_maybe_none` / `dropvalues*` therefore compile to the
        // plain `None` store with no collecting write-barrier slow path.
        if value.is_null() {
            unsafe { self.items_mut_ptr().add(index).write(value) };
            return;
        }
        // minimark.py:1063-1071 `writebarrier_before_copy` / the ordinary
        // `setarrayitem_gc` rewrite: inspect TRACK_YOUNG_PTRS inline and enter
        // the collecting slow path only while the old array still needs to be
        // remembered.  The barrier clears this bit, so every later store into
        // the same array is a plain write.  Nursery arrays and StdAlloc
        // fallback arrays also carry a zero flag word and take this arm.
        let header = unsafe { majit_gc::header::header_of(self as *mut Self as usize) };
        if unsafe { !(*header).has_flag(majit_gc::flags::TRACK_YOUNG_PTRS) } {
            unsafe { self.items_mut_ptr().add(index).write(value) };
            return;
        }
        let _roots = crate::gc_roots::push_roots();
        let root_base = _roots.base();
        _roots.pin_root(self as *mut Self as PyObjectRef);
        _roots.pin_root(value);
        let array = _roots.get(root_base) as *mut Self;
        // Every mutable FixedObjectArray has a real header word: managed frame
        // locals carry the collector's header, while the StdAlloc snapshot
        // fallback is deliberately prefixed with a zeroed one
        // (`alloc_fixed_array_with_header`).  This is therefore the ordinary
        // RPython `setarrayitem_gc` shape: test the header flag directly and
        // enter the membership-free slow path only when it is set.  The zeroed
        // fallback header makes the same call a no-op without an arena lookup.
        crate::gc_hook::try_gc_write_barrier_managed(array as *mut u8);
        // The barrier may wait behind a foreign collection. Reload the array
        // as well as the value before the store: RPython's setarrayitem_gc
        // keeps both live across the barrier.
        let array = _roots.get(root_base) as *mut Self;
        let value = _roots.get(root_base + 1);
        unsafe { (*array).items_mut_ptr().add(index).write(value) };
    }

    pub fn to_vec(&self) -> Vec<PyObjectRef> {
        self.as_slice().to_vec()
    }

    pub fn swap(&mut self, a: usize, b: usize) {
        self.as_mut_slice().swap(a, b);
    }
}

impl Index<usize> for FixedObjectArray {
    type Output = PyObjectRef;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl IndexMut<usize> for FixedObjectArray {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}

/// Allocate a fixed-length `FixedObjectArray` GcArray pre-filled from
/// `values`, for `W_TypeObject.mro_w` (typeobject.py:179 `mro_w?[*]`, an
/// immutable `[W_Root]`). The block is `PY_OBJECT_ARRAY_GC_TYPE_ID`
/// (`Ptr(GcArray(OBJECTPTR))`), so the collector reads its length from the
/// offset-0 header and forwards items[0..len]; the prepass types
/// `w_type_get_mro`'s result as `SomeList(SomeInstance(PyObjectRef))`.
///
/// Allocated on the **stable** (non-moving old-gen) path via
/// [`crate::gc_hook::try_gc_alloc_stable_raw`] — the owning `W_TypeObject`
/// is a `malloc_typed` Box-immortal whose custom trace never fires, so the
/// MRO block cannot be marked through its owner; keeping it old-gen means
/// the major collector always scans it from the prebuilt-root set (the
/// `walk_type_dicts_gc` mro forwarding), never minor-relocates it, and the
/// caller's `values` slice stays valid because the stable allocator does
/// not collect. Falls back to a header-prefixed `std::alloc` block when no
/// GC hook is installed (bootstrap / pure interpreter). A young MRO element
/// (a metaclass `mro()` returning fresh types) is registered on the
/// remembered set via the old→young write barrier.
pub unsafe fn alloc_mro_block_gc(values: &[PyObjectRef]) -> *mut FixedObjectArray {
    let len = values.len();
    let payload = FIXED_ARRAY_ITEMS_OFFSET + len * std::mem::size_of::<PyObjectRef>();
    let raw = crate::gc_hook::try_gc_alloc_stable_raw(PY_OBJECT_ARRAY_GC_TYPE_ID, payload);
    let block = if !raw.is_null() {
        raw as *mut FixedObjectArray
    } else {
        // Bootstrap / no-hook fallback: a plain `std::alloc` block. It carries
        // no GcHeader (the collector never observes it — `try_gc_owns_object`
        // reports false), matching the mapdict-storage fallback contract.
        let layout = Layout::from_size_align(payload, std::mem::align_of::<FixedObjectArray>())
            .expect("mro block layout");
        let mem = unsafe { alloc(layout) };
        if mem.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        mem as *mut FixedObjectArray
    };
    unsafe {
        (*block).len = len;
        let items = (*block).items_mut_ptr();
        for (i, &v) in values.iter().enumerate() {
            items.add(i).write(v);
        }
        // Old→young barrier if the block landed in old-gen and any element is a
        // young object: registers the block on the remembered set so a later
        // minor collection forwards the young MRO entries.
        if crate::gc_hook::try_gc_owns_object(block as *mut u8) {
            crate::gc_hook::try_gc_write_barrier(block as *mut u8);
        }
    }
    block
}

// ─── GcTypedArray: typed array helper for resume / blackhole ─────────
//
// llmodel.py:788-789: bh_new_array / bh_new_array_clear
// llmodel.py:607-619: bh_setarrayitem_gc_r/i/f, bh_getarrayitem_gc_r/i/f
// resume.py:1444-1537: ResumeDataDirectReader allocate_array + setarrayitem_*
//
// RPython GC arrays are flat varsize blocks:
//
//     [length: WORD] [item_0] [item_1] ... [item_(length-1)]
//
// `GcTypedArray` mirrors that storage shape so backend
// `bh_getarrayitem_gc_*` / `bh_setarrayitem_gc_*` pointer arithmetic can
// operate on the same object that resume materialization returns.  This
// is still an allocator adaptation: the block is allocated with
// `std::alloc::alloc_zeroed` rather than MiniMark's nursery, so GC
// tracing/write barriers are not claimed here.

/// Offset of the length prefix within `GcTypedArray`.
pub const GC_TYPED_ARRAY_LEN_OFFSET: usize = std::mem::offset_of!(GcTypedArray, len);

/// Offset of the first item within `GcTypedArray`.
///
/// No [`ArrayToken`] companion, and deliberately so: the item size comes from
/// the resume / blackhole array descriptor rather than from this block (one
/// struct standing in for the Ref / Int / Float / Struct ARRAY lltypes), and no
/// tid is registered for it — `allocate_flat_gc_typed_array` allocates through
/// `std::alloc::alloc_zeroed`, so the collector never reads a shape here. These
/// two offsets describe this struct's own pointer arithmetic only; they are not
/// a GcArray shape any tid may be registered with.
pub const GC_TYPED_ARRAY_ITEMS_OFFSET: usize = std::mem::offset_of!(GcTypedArray, items);

/// Typed varsize array helper used by the resume / blackhole readers.
/// The element type is carried by the array descriptor/API call, matching
/// RPython's typed `ArrayDescr`; the runtime block itself stores only the
/// length prefix followed by inline bytes.
#[repr(C)]
pub struct GcTypedArray {
    /// Length prefix. Backend `bh_arraylen_gc` reads this word at offset 0.
    pub len: usize,
    /// Flexible-array tail marker. Actual items follow immediately in
    /// memory and are sized by the allocation descriptor.
    items: [u8; 0],
}

/// Array element kind — resume.py:656 arraydescr.is_array_of_* / FLAG_STRUCT parity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArrayKind {
    Ref,
    Int,
    Float,
    /// Array(Struct(...)) — interior fields, item_size from arraydescr.
    Struct,
}

/// resume.py:1444-1447, llmodel.py:788-790 — API alias.
/// RPython: `bh_new_array_clear = bh_new_array` (llmodel.py:790).
/// Upstream both call `gc_malloc_array` which allocates a zero-filled
/// varsize block. Ref/int/float slots are word-sized.
pub fn allocate_array(length: usize, kind: ArrayKind, _clear: bool) -> *mut GcTypedArray {
    let item_size = match kind {
        ArrayKind::Ref | ArrayKind::Int | ArrayKind::Float => std::mem::size_of::<usize>(),
        ArrayKind::Struct => 0,
    };
    allocate_array_with_item_size(length, kind, item_size, _clear)
}

/// Same allocator as [`allocate_array`], but preserves the descriptor's
/// item size for array-of-structs materialization.
pub fn allocate_array_with_item_size(
    length: usize,
    _kind: ArrayKind,
    item_size: usize,
    _clear: bool,
) -> *mut GcTypedArray {
    allocate_flat_gc_typed_array(length, item_size)
}

/// resume.py:749 VArrayStructInfo.allocate — API alias.
/// Allocate a flat byte buffer for Array(Struct(...)):
/// `[len][num_elems * item_size bytes]`.
pub fn allocate_array_struct(num_elems: usize, item_size: usize) -> *mut GcTypedArray {
    allocate_flat_gc_typed_array(num_elems, item_size)
}

fn allocate_flat_gc_typed_array(length: usize, item_size: usize) -> *mut GcTypedArray {
    let layout = gc_typed_array_layout(length, item_size);
    unsafe {
        let raw = alloc_zeroed(layout);
        if raw.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let array = raw as *mut GcTypedArray;
        (*array).len = length;
        array
    }
}

fn gc_typed_array_layout(length: usize, item_size: usize) -> Layout {
    let items_size = length
        .checked_mul(item_size)
        .expect("GcTypedArray item bytes overflow");
    let total = GC_TYPED_ARRAY_ITEMS_OFFSET
        .checked_add(items_size)
        .expect("GcTypedArray allocation size overflow");
    Layout::from_size_align(total, std::mem::align_of::<GcTypedArray>())
        .expect("GcTypedArray layout")
}

/// Return the items base pointer of a `GcTypedArray`.
#[inline]
pub unsafe fn gc_typed_array_items_base(array: *mut GcTypedArray) -> *mut u8 {
    if array.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { (array as *mut u8).add(GC_TYPED_ARRAY_ITEMS_OFFSET) }
}

/// llmodel.py:596-619 bh_get/setarrayitem_gc_* access the raw byte
/// offset with no bounds check — a null array or out-of-range index
/// never occurs in correct jitcode. Panic loudly instead of reading
/// or writing out of bounds.
#[inline]
unsafe fn gc_typed_array_item_ptr(
    array: *mut GcTypedArray,
    index: usize,
    item_size: usize,
) -> *mut u8 {
    assert!(!array.is_null(), "gc_typed_array_item_ptr: null array");
    let len = unsafe { (*array).len };
    assert!(
        index < len,
        "gc_typed_array_item_ptr: index {index} out of range (len {len})"
    );
    unsafe { gc_typed_array_items_base(array).add(index * item_size) }
}

/// llmodel.py:607-609 bh_setarrayitem_gc_r parity.
pub fn setarrayitem_ref(array: *mut GcTypedArray, index: usize, value: PyObjectRef) {
    unsafe {
        let ptr = gc_typed_array_item_ptr(array, index, std::mem::size_of::<PyObjectRef>());
        (ptr as *mut PyObjectRef).write_unaligned(value);
    }
}

/// llmodel.py:596-598 bh_getarrayitem_gc_r parity.
/// Read a `PyObjectRef` from a ref array slot.
pub fn getarrayitem_ref(array: *const GcTypedArray, index: usize) -> PyObjectRef {
    unsafe {
        let ptr = gc_typed_array_item_ptr(
            array as *mut GcTypedArray,
            index,
            std::mem::size_of::<PyObjectRef>(),
        );
        (ptr as *const PyObjectRef).read_unaligned()
    }
}

/// llmodel.py:613-615 bh_setarrayitem_gc_i parity.
/// Write a raw i64 to an int array slot.
pub fn setarrayitem_int(array: *mut GcTypedArray, index: usize, value: i64) {
    unsafe {
        let ptr = gc_typed_array_item_ptr(array, index, std::mem::size_of::<i64>());
        (ptr as *mut i64).write_unaligned(value);
    }
}

/// llmodel.py:618-619 bh_setarrayitem_gc_f parity.
/// Write a raw f64 to a float array slot.
pub fn setarrayitem_float(array: *mut GcTypedArray, index: usize, value: f64) {
    unsafe {
        let ptr = gc_typed_array_item_ptr(array, index, std::mem::size_of::<f64>());
        (ptr as *mut f64).write_unaligned(value);
    }
}

/// resume.py:757 setinteriorfield(i, array, num, fielddescrs[j]) parity.
/// resume.py:1520-1529 ResumeDataDirectReader: dispatch on descr type.
/// llmodel.py:648-665: byte offset = elem_idx * item_size + field_offset.
///
pub fn setinteriorfield(
    array: *mut GcTypedArray,
    elem_idx: usize,
    field_offset: usize,
    field_size: usize,
    item_size: usize,
    descr_field_type: u8,
    value: i64,
) {
    if array.is_null() {
        return;
    }
    if item_size == 0 || field_size == 0 {
        return;
    }
    let len = gcarray_len(array);
    let Some(byte_offset) = elem_idx
        .checked_mul(item_size)
        .and_then(|base| base.checked_add(field_offset))
    else {
        return;
    };
    let Some(end) = byte_offset.checked_add(field_size) else {
        return;
    };
    let Some(total) = len.checked_mul(item_size) else {
        return;
    };
    if end > total {
        return;
    }
    unsafe {
        let ptr = gc_typed_array_items_base(array).add(byte_offset);
        match descr_field_type {
            2 => {
                let bits = value as u64;
                std::ptr::copy_nonoverlapping(bits.to_ne_bytes().as_ptr(), ptr, field_size.min(8));
            }
            0 => {
                let raw = value as usize;
                std::ptr::copy_nonoverlapping(
                    raw.to_ne_bytes().as_ptr(),
                    ptr,
                    field_size.min(std::mem::size_of::<usize>()),
                );
            }
            _ => {
                std::ptr::copy_nonoverlapping(value.to_ne_bytes().as_ptr(), ptr, field_size.min(8));
            }
        }
    }
}

/// Resume parity: get the length of a GcTypedArray.
pub fn gcarray_len(array: *const GcTypedArray) -> usize {
    if array.is_null() {
        return 0;
    }
    unsafe { (*array).len }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gc_typed_array_ref_roundtrip() {
        let arr = allocate_array(3, ArrayKind::Ref, true);
        assert_eq!(gcarray_len(arr), 3);
        // Zero-filled allocation: every slot reads back null.
        for i in 0..3 {
            assert!(getarrayitem_ref(arr, i).is_null());
        }
        let a = 0x1000usize as PyObjectRef;
        let b = 0x2008usize as PyObjectRef;
        setarrayitem_ref(arr, 0, a);
        setarrayitem_ref(arr, 2, b);
        assert_eq!(getarrayitem_ref(arr, 0), a);
        assert!(getarrayitem_ref(arr, 1).is_null());
        assert_eq!(getarrayitem_ref(arr, 2), b);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn gc_typed_array_out_of_range_read_panics() {
        let arr = allocate_array(3, ArrayKind::Ref, true);
        getarrayitem_ref(arr, 3);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn gc_typed_array_out_of_range_write_panics() {
        let arr = allocate_array(3, ArrayKind::Ref, true);
        setarrayitem_ref(arr, 3, std::ptr::null_mut());
    }
}
