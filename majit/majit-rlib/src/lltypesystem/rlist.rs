//! `rpython/rtyper/lltypesystem/rlist.py` — the low-level body a list is
//! lowered to.
//!
//! Only the scalar body lives here: a `Ptr(GcArray(Float))` /
//! `Ptr(GcArray(Signed))`, which is what `erase([float])` / `erase([int])`
//! produce for the Float / Integer list strategies (`listobject.py`
//! FloatListStrategy / IntegerListStrategy) and what `rbigint._digits` is
//! lowered to. An 8-byte capacity header (the GcArray length, rlist.py:251
//! `len(l.items)`) is followed by inline 8-byte items (f64 or i64). The live
//! list length lives on the enclosing wrapper (rlist.py:116
//! `("length", Signed)`), so the header is the allocated capacity, fixed for
//! the block's lifetime (a grow allocates a fresh block).
//!
//! Items are non-pointer words, so the collector has no inner refs to trace.
//!
//! The `GcArray(OBJECTPTR)` body (rlist.py:84) is not here: it is reachable
//! only from the object model that owns its item type.

use std::alloc::{Layout, alloc, alloc_zeroed};

/// GC type id for `Ptr(GcArray(Signed))`. This is a distinct ARRAY identity
/// from `GcArray(Float)` even though the collector trace shape is the same —
/// `GcLLDescr_framework.init_array_descr` gives each ARRAY lltype its own tid.
///
/// The value is the slot `gc.register_type` returns for it in pyre's
/// registration order; `pyre_object::object_array` re-exports it alongside the
/// rest of that numbering.
pub const GC_INT_ARRAY_GC_TYPE_ID: u32 = 41;

/// GC type id for `Ptr(GcArray(Float))` — the [`GC_INT_ARRAY_GC_TYPE_ID`] twin.
pub const GC_FLOAT_ARRAY_GC_TYPE_ID: u32 = 42;

/// Route items-block allocations through the moving nursery instead of
/// `std::alloc`. Read once; default ON — the nursery path mirrors the
/// `GcArray` body upstream allocates (rlist.py:84) and is validated identical
/// to the `std::alloc` fallback (check.py 158 both backends, both gate states;
/// fannkuch/nbody/spectral_norm timings unchanged). `PYRE_GC_ITEMSBLOCK=0`
/// (or `off`/`false`) restores the `std::alloc` fallback to bisect a
/// suspected block-GC regression.
///
/// Reads (and lazily initialises) the runtime-mutable `ENABLED` `OnceLock`,
/// not a build-time constant, so the JIT residualizes the call instead of
/// tracing into it. The `-> bool` return fits a single word and it cannot
/// raise.
///
/// This gates every items block, including the `GcArray(OBJECTPTR)` one that
/// lives with the object model — that block is the same lowering of the same
/// upstream file, so one gate answers for both.
#[majit_macros::dont_look_inside]
pub fn itemsblock_gc_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("PYRE_GC_ITEMSBLOCK")
            .map(|v| !matches!(v.trim(), "0" | "off" | "false" | ""))
            .unwrap_or(true)
    })
}

/// `#[repr(C)] { capacity, items: [u64; 0] }` — the `GcArray(Float)` /
/// `GcArray(Signed)` body. Layout: offset 0 = `capacity` (GcArray length
/// header), offset 8 = items[0..capacity]. Items are 8-byte words read as
/// `f64` / `i64` by the wrapper; the JIT-visible array descriptor carries the
/// element type.
#[repr(C)]
pub struct TypedItemsBlock {
    /// Allocated capacity — the GcArray length header (rlist.py:251).
    pub capacity: usize,
    /// Items inline after the header; size known only at allocation time.
    items: [u64; 0],
}

pub const TYPED_ITEMS_BLOCK_ITEMS_OFFSET: usize = std::mem::offset_of!(TypedItemsBlock, items);

/// Offset of the `capacity` header the collector reads as the GcArray length.
pub const TYPED_ITEMS_BLOCK_LEN_OFFSET: usize = std::mem::offset_of!(TypedItemsBlock, capacity);

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

pub fn typed_items_block_layout(cap: usize) -> Layout {
    try_typed_items_block_layout(cap).expect("TypedItemsBlock layout")
}

pub fn try_typed_items_block_layout(cap: usize) -> Option<Layout> {
    let items_size = cap.checked_mul(std::mem::size_of::<u64>())?;
    let total = TYPED_ITEMS_BLOCK_ITEMS_OFFSET.checked_add(items_size)?;
    Layout::from_size_align(total, std::mem::align_of::<TypedItemsBlock>()).ok()
}

/// Allocate a scalar GcArray in the no-collect moving nursery.
///
/// This is the `rbigint._digits` allocation shape. Bigint arithmetic may hold
/// several unboxed Rust `RBigInt` handles at once, so the allocator must not
/// trigger a collection while those raw digit pointers are live. The completed
/// result's digit edge is explicitly rooted when its RBigInt payload is boxed
/// by `alloc_rbigint_nursery_collecting`. Once a backend owns the heap,
/// allocation failure must remain a failure: a raw fallback would leave
/// `RBigInt._digits` pointing outside the managed heap even though its
/// descriptor traces that field as `GcArray(Signed)`.
pub unsafe fn alloc_typed_items_block_nursery(cap: usize, tid: u32) -> *mut TypedItemsBlock {
    unsafe {
        try_alloc_typed_items_block_nursery(cap, tid)
            .unwrap_or_else(|| std::alloc::handle_alloc_error(Layout::new::<TypedItemsBlock>()))
    }
}

/// Fallible companion of [`alloc_typed_items_block_nursery`].
///
/// `ll_newlist` (rlist.py:324-329) allocates the items array and writes the
/// length header; the items keep whatever the nursery held, because
/// `malloc_zero_filled` is false for incminimark (incminimark.py:211). Callers
/// that need `[0] * n` clear it themselves with [`typed_items_block_clear`],
/// which is what `ll_alloc_and_set` does (rtyper/rlist.py:494-503); callers
/// building a list display write every slot.
///
/// # Safety
/// `tid` must name a registered array type with no destructor and no weakref
/// flag — the `GcArray(Signed)` / `GcArray(Float)` bodies this serves. Every
/// item the caller reads must be one it has written.
pub unsafe fn try_alloc_typed_items_block_nursery(
    cap: usize,
    tid: u32,
) -> Option<*mut TypedItemsBlock> {
    let cap = cap.max(1);
    let layout = try_typed_items_block_layout(cap)?;
    // Nothing owns the heap during a bare unit test or the pre-`init_gc_subsystem`
    // bootstrap, and there the `std::alloc` path below is the whole heap. Once a
    // backend is installed a null is the GC refusing, and the caller must see
    // that rather than a block outside the managed heap
    // (`majit_gc::gc_allocator_installed`).
    if itemsblock_gc_enabled() && majit_gc::gc_allocator_installed() {
        // `GcArray(Signed)` / `GcArray(Float)` bodies: no finalizer, not a
        // WEAKREF, so `gct_fv_gc_malloc` (`framework.py:820-838`) reaches
        // `malloc_fast`.
        let raw = unsafe { majit_gc::alloc_fast_nursery_typed(tid, layout.size()) }.0 as *mut u8;
        if raw.is_null() {
            return None;
        }
        let block = raw as *mut TypedItemsBlock;
        unsafe { (*block).capacity = cap };
        return Some(block);
    }
    unsafe {
        let raw = alloc(layout);
        if raw.is_null() {
            return None;
        }
        let block = raw as *mut TypedItemsBlock;
        (*block).capacity = cap;
        Some(block)
    }
}

/// `rgc.ll_arrayclear(l.ll_items())` — zero every item of a freshly allocated
/// block, the second half of `ll_alloc_and_set(LIST, count, 0)`
/// (rtyper/rlist.py:494-503).
///
/// # Safety
/// `block` must be a live items block.
#[inline]
pub unsafe fn typed_items_block_clear(block: *mut TypedItemsBlock) {
    unsafe {
        std::ptr::write_bytes(
            typed_items_block_items_base(block),
            0,
            (*block).capacity * std::mem::size_of::<u64>(),
        );
    }
}

/// The `[Digit]` list body `rbigint._digits` is lowered to — a
/// `Ptr(GcArray(Signed))`.
///
/// This is a namespace, not a wrapper: the field stays a raw
/// `*mut TypedItemsBlock` so the annotator keeps seeing the source-level
/// `GcArray(Signed)` shape and `RBigInt::digit` keeps projecting `_digits`
/// directly.
///
/// What it owns is the *vocabulary*. Upstream draws this line already:
/// `rbigint.py` writes `[NULLDIGIT] * size` and `rbigint(digits, sign, size)`
/// and never names a malloc, an allocation tier, or an array type id — the
/// rtyper supplies those, `ll_newlist` (rlist.py:324-329) and
/// `ll_alloc_and_set` (rtyper/rlist.py:494-503). A consumer of this type
/// spells the list operation; the array's GC type id and the tier it is
/// allocated in stay here.
pub struct Digits;

impl Digits {
    /// `ll_newlist(LIST, length)` (rlist.py:324-329) — allocate the items
    /// array without initializing it. `malloc_zero_filled` is false for
    /// incminimark (incminimark.py:211), so the items keep whatever the
    /// nursery held.
    ///
    /// # Safety
    /// Every item the caller reads must be one it has written. Use
    /// [`Digits::alloc_and_set_zero`] for the `[NULLDIGIT] * n` shape.
    #[inline]
    pub unsafe fn new(length: usize) -> *mut TypedItemsBlock {
        unsafe { alloc_typed_items_block_nursery(length, GC_INT_ARRAY_GC_TYPE_ID) }
    }

    /// Fallible [`Digits::new`], for the upstream paths whose translated
    /// allocation carries an explicit `MemoryError` edge.
    ///
    /// # Safety
    /// Same obligation as [`Digits::new`].
    #[inline]
    pub unsafe fn try_new(length: usize) -> Option<*mut TypedItemsBlock> {
        unsafe { try_alloc_typed_items_block_nursery(length, GC_INT_ARRAY_GC_TYPE_ID) }
    }

    /// `ll_alloc_and_set(LIST, count, 0)` (rtyper/rlist.py:494-503) —
    /// `ll_newlist` followed by `rgc.ll_arrayclear`. This is what
    /// `[NULLDIGIT] * count` lowers to.
    ///
    /// # Safety
    /// The result is a fresh block outside any traced owner; the caller must
    /// store it into one before anything can collect.
    #[inline]
    pub unsafe fn alloc_and_set_zero(count: usize) -> *mut TypedItemsBlock {
        unsafe {
            let block = Self::new(count);
            typed_items_block_clear(block);
            block
        }
    }

    /// Fallible [`Digits::alloc_and_set_zero`].
    ///
    /// # Safety
    /// Same obligation as [`Digits::alloc_and_set_zero`].
    #[inline]
    pub unsafe fn try_alloc_and_set_zero(count: usize) -> Option<*mut TypedItemsBlock> {
        unsafe {
            let block = Self::try_new(count)?;
            typed_items_block_clear(block);
            Some(block)
        }
    }

    /// The one-element body of a prebuilt list, at process lifetime.
    ///
    /// Upstream's prebuilt `rbigint` digit lists are translated constants with
    /// the same lifetime; see [`alloc_typed_items_block_immortal`].
    ///
    /// # Safety
    /// Only for a prebuilt whose owner is itself immortal.
    #[inline]
    pub unsafe fn prebuilt(item: i64) -> *mut TypedItemsBlock {
        unsafe {
            let block = alloc_typed_items_block_immortal(1);
            *(typed_items_block_items_base(block) as *mut i64) = item;
            block
        }
    }
}

/// Allocate a headerless, process-lifetime `GcArray(Signed/Float)` body.
///
/// This is only for translated prebuilt objects whose module-global owner is
/// itself immortal (not for ordinary arrays). The layout remains the exact
/// `[length][items...]` GcArray shape seen by generated code, while the hybrid
/// collector deliberately treats the address as non-managed. Upstream's
/// prebuilt `rbigint` digit lists have the same process lifetime.
pub unsafe fn alloc_typed_items_block_immortal(cap: usize) -> *mut TypedItemsBlock {
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
