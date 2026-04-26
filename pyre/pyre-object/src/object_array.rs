use std::alloc::{Layout, alloc, dealloc};
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
/// Task #141 GC-root infrastructure + Drop source-of-truth
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
/// `w_float_new`. Captured in Task #98 / `l1_step4ab*` memory.
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

// ─── GcArray: RPython GC array allocation + setarrayitem parity ──────────
//
// resume.py:1444-1537 ResumeDataDirectReader:
//   allocate_array(length, arraydescr, clear) → GCREF
//   setarrayitem_ref(array, index, fieldnum, arraydescr) → write decoded ref
//   setarrayitem_int(array, index, fieldnum, arraydescr) → write decoded int
//   setarrayitem_float(array, index, fieldnum, arraydescr) → write decoded float
//
// In pyre, GcArray is a boxed PyObjectArray on the heap. The returned
// pointer is a raw `*mut PyObjectArray` cast to usize for GcRef.

// ─── GcTypedArray: typed array helper for resume / blackhole ─────────
//
// llmodel.py:788-789: bh_new_array / bh_new_array_clear
// llmodel.py:607-619: bh_setarrayitem_gc_r/i/f, bh_getarrayitem_gc_r/i/f
// resume.py:1444-1537: ResumeDataDirectReader allocate_array + setarrayitem_*
//
// RPython GC arrays are typed: ref[], int[], float[]. Each slot stores
// a raw value of the corresponding type. pyre's `GcTypedArray` keeps
// the typed distinction at the API level but DOES NOT match upstream's
// memory shape:
//
// **NON-PARITY (PRE-EXISTING-ADAPTATION).** Upstream `gc_malloc_array`
// (`framework.py:837-849 gct_fv_gc_malloc_varsize`) is a *varsize*
// allocation parameterised by `(length, basesize, itemsize,
// length_offset)`, producing a single flat block:
//
//     [length: WORD] [item_0] [item_1] ... [item_(length-1)]
//
// `GcTypedArray` instead heap-boxes a fixed-size Rust enum that wraps
// a `Vec<T>`, giving a double indirection
// (`*mut GcTypedArray -> Vec.ptr -> items`). The Vec storage lives on
// `std::alloc`'s heap, completely outside any GC nursery, so the GC
// walker has no path to the inline `PyObjectRef`s in `Ref(_)` /
// `Struct { data, .. }`.
//
// Convergence target: replace each variant with a flat varsize struct
// matching upstream — see [`ItemsBlock`] above for the exact shape
// (`rpython/rtyper/lltypesystem/rlist.py:84 GcArray(OBJECTPTR)`) and
// register through `TypeInfo::varsize(basesize, itemsize,
// length_offset, items_have_gc_ptrs, gc_ptr_offsets)` matching
// `framework.py:839`. Allocation site (`allocate_array` /
// `allocate_array_struct`) routes through `malloc_raw` rather than
// `malloc` to avoid claiming GC-managed semantics it does not yet
// honour. The Vec contents are reachable only through the resume /
// blackhole walker holding `*mut GcTypedArray` directly as a root
// (resume.py:1444-1537 reader pattern).

/// Typed array helper used by the resume / blackhole readers.
/// `descr.py:273 ArrayDescr.flag` parity at the API level only —
/// the memory shape is NOT upstream-equivalent (see module-level
/// comment).
pub enum GcTypedArray {
    /// FLAG_POINTER: each slot is a PyObjectRef (GCREF).
    Ref(Vec<PyObjectRef>),
    /// FLAG_SIGNED/FLAG_UNSIGNED: each slot is a raw i64.
    Int(Vec<i64>),
    /// FLAG_FLOAT: each slot is a raw f64.
    Float(Vec<f64>),
    /// FLAG_STRUCT: Array(Struct(...)) — flat byte buffer.
    /// llmodel.py:648-665 bh_setinteriorfield_gc_* parity.
    /// Layout: num_elems elements, each item_size bytes, stored inline.
    /// Access: elem_idx * item_size + field_offset.
    Struct {
        item_size: usize,
        num_elems: usize,
        data: Vec<u8>,
    },
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
/// Upstream both call `gc_malloc_array` which allocates a varsize
/// block from the GC nursery (always zero-filled).
///
/// **NON-PARITY** at the storage level: pyre boxes a fixed-size
/// `GcTypedArray` enum (`malloc_raw`) whose `Vec` payload lives on
/// `std::alloc`'s heap. The result is invisible to the GC walker
/// — see the module-level comment for the convergence path through
/// flat varsize blocks à la [`ItemsBlock`].
pub fn allocate_array(length: usize, kind: ArrayKind, _clear: bool) -> *mut GcTypedArray {
    // llmodel.py:790: bh_new_array_clear = bh_new_array.
    // Both allocate from gc_malloc_array (always zero-filled).
    let arr = match kind {
        ArrayKind::Ref => GcTypedArray::Ref(vec![PY_NULL; length]),
        ArrayKind::Int => GcTypedArray::Int(vec![0i64; length]),
        ArrayKind::Float => GcTypedArray::Float(vec![0.0f64; length]),
        ArrayKind::Struct => GcTypedArray::Struct {
            item_size: 0,
            num_elems: length,
            data: Vec::new(),
        },
    };
    crate::lltype::malloc_raw(arr)
}

/// resume.py:749 VArrayStructInfo.allocate — API alias.
/// Allocate a flat byte buffer for Array(Struct(...)).
/// Layout: num_elems elements × item_size bytes, zero-filled.
/// llmodel.py: `gc_malloc_array(basesize + num_elems * itemsize)`.
///
/// **NON-PARITY** at the storage level: same caveat as
/// [`allocate_array`] above. The byte buffer lives on `std::alloc`,
/// not in the GC nursery, so reads/writes through
/// [`setinteriorfield`] do not engage write-barriers. Until the
/// flat-varsize port lands, callers must hold `*mut GcTypedArray` as
/// an explicit root for any contained PyObjectRefs to survive a
/// collection.
pub fn allocate_array_struct(num_elems: usize, item_size: usize) -> *mut GcTypedArray {
    let total_bytes = num_elems * item_size;
    let arr = GcTypedArray::Struct {
        item_size,
        num_elems,
        data: vec![0u8; total_bytes],
    };
    crate::lltype::malloc_raw(arr)
}

/// llmodel.py:607-609 bh_setarrayitem_gc_r parity.
pub fn setarrayitem_ref(array: *mut GcTypedArray, index: usize, value: PyObjectRef) {
    if array.is_null() {
        return;
    }
    let arr = unsafe { &mut *array };
    if let GcTypedArray::Ref(v) = arr {
        if index < v.len() {
            v[index] = value;
        }
    }
}

/// llmodel.py:613-615 bh_setarrayitem_gc_i parity.
/// Write a raw i64 to an int array slot.
pub fn setarrayitem_int(array: *mut GcTypedArray, index: usize, value: i64) {
    if array.is_null() {
        return;
    }
    let arr = unsafe { &mut *array };
    match arr {
        GcTypedArray::Int(v) => {
            if index < v.len() {
                v[index] = value;
            }
        }
        // Fallback: box as W_IntObject into ref array
        GcTypedArray::Ref(v) => {
            if index < v.len() {
                v[index] = crate::intobject::w_int_new(value);
            }
        }
        _ => {}
    }
}

/// llmodel.py:618-619 bh_setarrayitem_gc_f parity.
/// Write a raw f64 to a float array slot.
pub fn setarrayitem_float(array: *mut GcTypedArray, index: usize, value: f64) {
    if array.is_null() {
        return;
    }
    let arr = unsafe { &mut *array };
    match arr {
        GcTypedArray::Float(v) => {
            if index < v.len() {
                v[index] = value;
            }
        }
        // Fallback: box as W_FloatObject into ref array
        GcTypedArray::Ref(v) => {
            if index < v.len() {
                v[index] = crate::floatobject::w_float_new(value);
            }
        }
        _ => {}
    }
}

/// resume.py:757 setinteriorfield(i, array, num, fielddescrs[j]) parity.
/// resume.py:1520-1529 ResumeDataDirectReader: dispatch on descr type.
/// llmodel.py:648-665: byte offset = elem_idx * item_size + field_offset.
///
/// For GcTypedArray::Struct: writes directly to the flat byte buffer.
/// For legacy Ref/Int/Float arrays: falls back to flat index computation.
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
    let arr = unsafe { &mut *array };
    match arr {
        GcTypedArray::Struct {
            data,
            item_size: is,
            ..
        } => {
            // llmodel.py:648-665 parity: byte_offset = elem_idx * item_size + field_offset
            let byte_offset = elem_idx * *is + field_offset;
            let end = byte_offset + field_size.min(8);
            if end <= data.len() {
                match descr_field_type {
                    2 => {
                        // bh_setinteriorfield_gc_f: write f64
                        let bits = value as u64;
                        data[byte_offset..byte_offset + 8.min(field_size)]
                            .copy_from_slice(&bits.to_ne_bytes()[..8.min(field_size)]);
                    }
                    0 => {
                        // bh_setinteriorfield_gc_r: write ref (pointer)
                        let ptr = value as usize;
                        let sz = std::mem::size_of::<usize>().min(field_size);
                        data[byte_offset..byte_offset + sz]
                            .copy_from_slice(&ptr.to_ne_bytes()[..sz]);
                    }
                    _ => {
                        // bh_setinteriorfield_gc_i: write int
                        let sz = field_size.min(8);
                        data[byte_offset..byte_offset + sz]
                            .copy_from_slice(&value.to_ne_bytes()[..sz]);
                    }
                }
            }
        }
        _ => {
            // Legacy fallback: flat index for Ref/Int/Float arrays.
            let fields_per_elem = if item_size > 0 { item_size } else { 1 };
            let flat = elem_idx * fields_per_elem + field_offset;
            match descr_field_type {
                2 => setarrayitem_float(array, flat, f64::from_bits(value as u64)),
                1 => setarrayitem_int(array, flat, value),
                _ => setarrayitem_ref(array, flat, value as PyObjectRef),
            }
        }
    }
}

/// Resume parity: get the length of a GcTypedArray.
pub fn gcarray_len(array: *const GcTypedArray) -> usize {
    if array.is_null() {
        return 0;
    }
    let arr = unsafe { &*array };
    match arr {
        GcTypedArray::Ref(v) => v.len(),
        GcTypedArray::Int(v) => v.len(),
        GcTypedArray::Float(v) => v.len(),
        GcTypedArray::Struct { num_elems, .. } => *num_elems,
    }
}
