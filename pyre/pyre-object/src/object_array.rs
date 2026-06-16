use std::alloc::{Layout, alloc, alloc_zeroed, dealloc};
use std::ops::{Index, IndexMut};

use crate::{PY_NULL, PyObjectRef};

/// GC type id for the variable-length backing block of
/// `W_ListObject.items` / `W_TupleObject.wrappeditems` /
/// `DictStorage.values`. Shape matches RPython's
/// `GcArray(OBJECTPTR)` from `rpython/rtyper/lltypesystem/rlist.py:84,116`
/// — a `T_IS_VARSIZE` block with an 8-byte single-slot `capacity`
/// header followed by inline `PyObjectRef` items. Registered with
/// `TypeInfo::varsize(8, 8, 0, items_have_gc_ptrs=true, [])` so the
/// GC walks each item slot as a Ref. Re-exported from
/// `pyre_jit_trace::descr` for existing call sites.
pub const PY_OBJECT_ARRAY_GC_TYPE_ID: u32 = 9;

/// GC type id for `Ptr(GcArray(Signed))` blocks materialized by resume /
/// blackhole paths. This is a distinct ARRAY identity from
/// `GcArray(Float)` even though the collector trace shape is the same.
/// Registered after the dictview (39) and getset-property (40) slots
/// so `gc.register_type` returns 41 here.
pub const GC_INT_ARRAY_GC_TYPE_ID: u32 = 41;

/// GC type id for `Ptr(GcArray(Float))` blocks materialized by resume /
/// blackhole paths. RPython gives each ARRAY lltype its own tid via
/// `GcLLDescr_framework.init_array_descr`.
pub const GC_FLOAT_ARRAY_GC_TYPE_ID: u32 = 42;

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
/// STEPPING-STONE (metadata precedes runtime, Phase L2 pending).
/// The header layout already matches upstream but the allocator
/// does NOT: `alloc_items_block` / `grow_items_block` below still
/// use `std::alloc::alloc`, not
/// `MiniMarkGC::alloc_varsize_typed(PY_OBJECT_ARRAY_GC_TYPE_ID,
/// cap)`. Until Phase L2 cuts the allocator over (blocked on
/// GC-root infrastructure + Drop source-of-truth
/// decision), the matching `PY_OBJECT_ARRAY_GC_TYPE_ID` and
/// `W_LIST_GC_TYPE_ID.gc_ptr_offsets = [offset_of!(items)]` are
/// inactive at collection time — the walker rejects the
/// non-nursery block pointer (collector.rs:377). Phase L1 of the
/// epic already landed: `W_ListObject` / `W_TupleObject` hold
/// `{length: usize, items: *mut ItemsBlock}` fields directly
/// (no more `PyObjectArray` fat wrapper for list/tuple).
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
/// existing items from `old`, NULL-initialising the rest, and
/// deallocating `old`. Returns the new block. `old` may be null
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

/// Allocate a fresh `ItemsBlock` with the given capacity.
///
/// STEPPING-STONE: still uses `std::alloc::alloc`. The
/// `try_gc_alloc_stable` migration was attempted but routes the
/// per-iteration list allocations through MiniMark's old-gen
/// (mark-sweep, non-moving), which regresses bench (cranelift
/// fannkuch timeout, dynasm nbody timeout) — old-gen-only
/// containers accumulate until major GC fires. The correct
/// long-term path is a nursery allocation behind caller-side
/// root tracking; the `try_gc_owns_object` infra in
/// `gc_hook.rs` is in place to support that follow-up. See
/// `40d4a041d7` docstring for the same finding on `w_int_new`/
/// `w_float_new`.
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
/// [`alloc_items_block`] or [`grow_items_block`]. STEPPING-STONE:
/// still bound to the matching `std::alloc` allocator above. Once
/// the alloc path migrates, this should discriminate via
/// `crate::gc_hook::try_gc_owns_object` so GC-managed blocks are
/// left for the major sweep instead of being dealloc'd directly.
unsafe fn dealloc_items_block(block: *mut ItemsBlock) {
    if block.is_null() {
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
        if !old.is_null() {
            dealloc_items_block(old);
        }
        fresh
    }
}

// ─── TypedItemsBlock: GcArray(Float)/GcArray(Signed) backing block ───────
//
// The Float / Integer list strategies (`listobject.py` FloatListStrategy /
// IntegerListStrategy) store their unboxed items in a `Ptr(GcArray(Float))` /
// `Ptr(GcArray(Signed))` — `erase([float])` / `erase([int])`. This block
// mirrors that shape: an 8-byte capacity header (the GcArray length, rlist.py:251
// `len(l.items)`) followed by inline 8-byte items (f64 or i64). The live list
// length lives on the enclosing `FloatArray` / `IntArray` wrapper (rlist.py:116
// `("length", Signed)`), so the header is the allocated capacity, fixed for the
// block's lifetime (a `grow` allocates a fresh block).
//
// Items are non-pointer words, so the GC walker has no inner refs to trace —
// unlike `ItemsBlock` (`GcArray(OBJECTPTR)`). STEPPING-STONE: like `ItemsBlock`,
// allocation still uses `std::alloc` rather than MiniMark's nursery; the
// matching `GC_FLOAT_ARRAY_GC_TYPE_ID` / `GC_INT_ARRAY_GC_TYPE_ID` are inactive
// at collection time until the Phase L2 allocator cutover.

/// `#[repr(C)] { capacity, items: [u64; 0] }` — the `GcArray(Float)` /
/// `GcArray(Signed)` body backing `FloatArray` / `IntArray`. Layout: offset 0 =
/// `capacity` (GcArray length header), offset 8 = items[0..capacity]. Items are
/// 8-byte words read as `f64` / `i64` by the wrapper; the JIT-visible array
/// descriptor carries the element type.
#[repr(C)]
pub struct TypedItemsBlock {
    /// Allocated capacity — the GcArray length header (rlist.py:251).
    pub capacity: usize,
    /// Items inline after the header; size known only at allocation time.
    items: [u64; 0],
}

pub const TYPED_ITEMS_BLOCK_ITEMS_OFFSET: usize = std::mem::offset_of!(TypedItemsBlock, items);

/// Items base pointer (`&items[0]`) of a `TypedItemsBlock`. Null-safe.
#[inline]
pub unsafe fn typed_items_block_items_base(block: *mut TypedItemsBlock) -> *mut u8 {
    if block.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { (block as *mut u8).add(TYPED_ITEMS_BLOCK_ITEMS_OFFSET) }
}

/// Allocated capacity (GcArray length header) of a `TypedItemsBlock`. 0 for null.
#[inline]
pub unsafe fn typed_items_block_capacity(block: *mut TypedItemsBlock) -> usize {
    if block.is_null() {
        return 0;
    }
    unsafe { (*block).capacity }
}

fn typed_items_block_layout(cap: usize) -> Layout {
    let total = TYPED_ITEMS_BLOCK_ITEMS_OFFSET + cap * std::mem::size_of::<u64>();
    Layout::from_size_align(total, std::mem::align_of::<TypedItemsBlock>())
        .expect("TypedItemsBlock layout")
}

/// Allocate a fresh zero-filled `TypedItemsBlock` with the given capacity.
/// Zero-fill matches `gc_malloc_array` (rlist.py:262-267 `_ll_list_resize_really`)
/// and the Float/Int strategy `_none_value` (0.0 / 0). `cap` is clamped to at
/// least 1 (rlist.py:251 overallocation policy keeps a slot for in-place growth).
pub unsafe fn alloc_typed_items_block(cap: usize) -> *mut TypedItemsBlock {
    let cap = cap.max(1);
    let layout = typed_items_block_layout(cap);
    unsafe {
        let raw = alloc_zeroed(layout);
        if raw.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let block = raw as *mut TypedItemsBlock;
        (*block).capacity = cap;
        block
    }
}

/// Grow a `TypedItemsBlock` to `new_cap`, copying `live_len` words from `old`,
/// zero-filling the rest, and deallocating `old`. `old` may be null.
/// rlist.py:262-267 parity.
pub unsafe fn grow_typed_items_block(
    old: *mut TypedItemsBlock,
    new_cap: usize,
    live_len: usize,
) -> *mut TypedItemsBlock {
    unsafe {
        let fresh = alloc_typed_items_block(new_cap);
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
        fresh
    }
}

/// Deallocate a `TypedItemsBlock`. No-op on null.
pub unsafe fn dealloc_typed_items_block(block: *mut TypedItemsBlock) {
    if block.is_null() {
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

/// Offset of the length prefix within `FixedObjectArray` (always 0).
pub const FIXED_ARRAY_LEN_OFFSET: usize = 0;

/// Offset of the first item within `FixedObjectArray` (immediately after
/// the length prefix). The JIT-visible array descriptor uses this as
/// `base_size` so `GETARRAYITEM_GC_*` reads items directly.
pub const FIXED_ARRAY_ITEMS_OFFSET: usize = std::mem::size_of::<usize>();

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

/// Offset of the length prefix within `GcTypedArray` (always 0).
pub const GC_TYPED_ARRAY_LEN_OFFSET: usize = 0;

/// Offset of the first item within `GcTypedArray`.
pub const GC_TYPED_ARRAY_ITEMS_OFFSET: usize = std::mem::size_of::<usize>();

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
