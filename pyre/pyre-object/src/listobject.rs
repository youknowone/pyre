//! W_ListObject — Python `list` with a minimal PyPy-style strategy split.
//!
//! Homogeneous integer and float lists keep unboxed storage, matching PyPy's
//! `IntegerListStrategy` / `FloatListStrategy` direction. Mixed lists fall back
//! to object storage.
//! The JIT's current raw-array fast path only handles object storage.

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)]

use crate::object_array::{
    ItemsBlock, TypedItemsBlock, alloc_list_items_block_gc, alloc_typed_items_block,
    dealloc_list_items_block, gc_int_array_gc_type_id, grow_list_items_block_gc,
    items_block_capacity, items_block_items_base, typed_items_block_items_base,
};
use crate::pyobject::*;
use crate::{
    FloatArray, IntArray,
    bytes_array::BytesArray,
    bytesobject::{BYTES_TYPE, w_bytes_block, w_bytes_from_block},
    floatobject::w_float_get_value,
    floatobject::w_float_new,
    intobject::{w_int_get_value, w_int_new},
    longobject::jit_bigint_to_i64_value,
    longobject::w_long_fits_int,
    longobject::w_long_get_value,
    tupleobject::is_plain_float_strict,
    unicode_array::UnicodeArray,
    unicodeobject::{w_str_from_storage, w_str_is_ascii, w_str_storage},
};
use std::cell::UnsafeCell;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicUsize, Ordering};

// PyPy's list strategy/storage transitions are indivisible under its GIL.
// Pyre keeps the list itself as the sole semantic owner and uses only narrow
// address-striped reentrant synchronization around those transitions.
struct ForkListLock(UnsafeCell<parking_lot::ReentrantMutex<()>>);
unsafe impl Sync for ForkListLock {}

impl ForkListLock {
    fn new() -> Self {
        Self(UnsafeCell::new(parking_lot::ReentrantMutex::new(())))
    }

    fn get(&self) -> &parking_lot::ReentrantMutex<()> {
        unsafe { &*self.0.get() }
    }

    unsafe fn reinit_after_fork(&self) {
        unsafe { self.0.get().write(parking_lot::ReentrantMutex::new(())) };
    }
}

static LIST_LOCKS: LazyLock<Vec<ForkListLock>> =
    LazyLock::new(|| (0..256).map(|_| ForkListLock::new()).collect());

type ListGuard = parking_lot::lock_api::ReentrantMutexGuard<
    'static,
    parking_lot::RawMutex,
    parking_lot::RawThreadId,
    (),
>;

/// Only the acquire itself is opaque to the tracer; the guard-holding bodies
/// stay look-inside, the same split `w_dict_lock` uses. A `dont_look_inside`
/// data function is excluded from the jitcode pipeline entirely — the
/// codewriter roots jitdriver portals and reaches everything else through
/// look-inside calls — so the tracer emits a residual call for the whole
/// operation instead of specializing the strategy dispatch.
#[majit_macros::dont_look_inside]
unsafe fn w_list_lock(obj: PyObjectRef) -> ListGuard {
    // A nursery list can move while its guard is held.  Stripe on its stable
    // class identity, not the movable instance address, so every operation on
    // one list continues to acquire the same lock after collection.
    let w_class = (*obj).w_class;
    let lock = LIST_LOCKS[(w_class as usize >> 4) & (LIST_LOCKS.len() - 1)].get();
    if let Some(guard) = lock.try_lock() {
        return guard;
    }
    let blocked = majit_gc::gc_sync::before_external_block();
    let guard = lock.lock();
    drop(blocked);
    guard
}

pub fn list_locks_after_fork_child() {
    for lock in LIST_LOCKS.iter() {
        unsafe { lock.reinit_after_fork() };
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ListStrategy {
    Object = 0,
    Integer = 1,
    Float = 2,
    /// listobject.py EmptyListStrategy — newly created or cleared list
    /// without any storage yet. First append picks a typed strategy via
    /// switch_to_correct_strategy.
    Empty = 3,
    /// listobject.py IntOrFloatListStrategy.  Entries share the
    /// `int_items` signed-longlong array: int32 values use RPython's
    /// 0xfffffffe NaN payload and floats keep their raw IEEE-754 bits.
    IntOrFloat = 4,
    /// listobject.py BytesListStrategy — erased `[rpython str]` payloads.
    Bytes = 5,
    /// listobject.py AsciiListStrategy — erased UTF-8 `[rpython str]`
    /// payloads, restricted to exact ASCII strings.
    Ascii = 6,
    /// listobject.py SizeListStrategy — empty storage carrying the allocation
    /// hint used when the first item selects a concrete strategy.
    Size = 7,
    /// listobject.py SimpleRangeListStrategy — the immutable storage tuple
    /// contains only the positive length for the sequence 0..length.
    SimpleRange = 8,
    /// listobject.py RangeListStrategy — the immutable storage tuple contains
    /// start, step, and positive length.
    Range = 9,
}

impl ListStrategy {
    /// `interp_magic.py strategy` reads the concrete strategy class name.
    /// Keep the spelling beside the representation discriminant so adding a
    /// PyPy list strategy cannot silently leave the diagnostic surface stale.
    #[inline]
    pub const fn class_name(self) -> &'static str {
        match self {
            Self::Object => "ObjectListStrategy",
            Self::Integer => "IntegerListStrategy",
            Self::Float => "FloatListStrategy",
            Self::Empty => "EmptyListStrategy",
            Self::IntOrFloat => "IntOrFloatListStrategy",
            Self::Bytes => "BytesListStrategy",
            Self::Ascii => "AsciiListStrategy",
            Self::Size => "SizeListStrategy",
            Self::SimpleRange => "SimpleRangeListStrategy",
            Self::Range => "RangeListStrategy",
        }
    }
}

/// Python list object.
///
/// Layout matches upstream `rpython/rtyper/lltypesystem/rlist.py:116`
/// `GcStruct("list", ("length", Signed), ("items",
/// Ptr(GcArray(OBJECTPTR))))`. The JIT parity-field pair is
/// `(length, items)`; `items` points at an `ItemsBlock` whose
/// offset-0 header holds the allocated capacity
/// (upstream `len(l.items)` per rlist.py:251).
///
/// `strategy`, `int_items`, `float_items`, `bytes_items`, `ascii_items`
/// implement PyPy's list
/// strategy split (`pypy/objspace/std/listobject.py`). Only the Object strategy
/// reads/writes `length` + `items`; Integer/IntOrFloat/Float/Bytes strategies
/// operate on their own typed arrays and keep `length = 0`, `items = null`.
#[repr(C)]
pub struct W_ListObject {
    pub ob_header: PyObject,
    /// CPython 3.14 `PyListObject.allocated`. The strategy backings use
    /// RPython arrays whose physical growth policy differs, while
    /// `list.__sizeof__()` exposes this logical pointer-slot allocation.
    pub allocated: isize,
    /// Live length under the Object strategy. Upstream `l.length`
    /// (rlist.py:116). Under Integer/Float strategies this mirrors
    /// `int_items.len()` / `float_items.len()` only when a strategy
    /// switch rewrites both together — typed-strategy operations do
    /// NOT update this field. Callers must read length via
    /// `w_list_len()` which dispatches on strategy.
    ///
    /// Read WITHOUT the list lock, so the slot is an atomic rather than a plain
    /// `usize`.  `Include/cpython/listobject.h PyList_GET_SIZE` answers a length
    /// under `Py_GIL_DISABLED` as `_Py_atomic_load_ssize_relaxed(&ob_size)` — a
    /// relaxed atomic load and no critical section — so a reader is entitled to
    /// a value from either side of a concurrent mutation but never to a torn
    /// one.  The JIT's `len` fold lowers to exactly that load through
    /// `list_length_descr`, and the mutators below write it while a compiled
    /// loop is reading; only an atomic makes that pair defined.  The writes are
    /// relaxed stores for the same reason — `&mut self` bounds no raw-pointer
    /// reader, so `get_mut()` would put the plain store straight back.
    ///
    /// Same size and bit validity as `usize`, so `offset_of!` and the JIT's
    /// `Type::Int` read are unchanged.
    pub length: AtomicUsize,
    /// `Ptr(GcArray(OBJECTPTR))` — rlist.py:116 `l.items`. Points at
    /// the `ItemsBlock` whose offset-0 header is the allocated
    /// capacity (= upstream `len(l.items)` per rlist.py:251). Null
    /// when the list is in a non-Object strategy (Empty/Integer/
    /// IntOrFloat/Float/Bytes); lazily allocated on strategy switch.
    /// SizeListStrategy instead uses this otherwise inactive traced slot for
    /// its shared strategy-state box, matching `EmptyListStrategy.clone`
    /// retaining the same `SizeListStrategy` instance.
    pub items: *mut ItemsBlock,
    pub strategy: ListStrategy,
    pub int_items: IntArray,
    pub float_items: FloatArray,
    pub bytes_items: BytesArray,
    pub ascii_items: UnicodeArray,
    /// PyPy `BaseUserClassMapdict` indexed instance storage for a native
    /// `list` subclass declaring `__slots__`.  Kept on the object itself,
    /// just like `W_UnicodeObject.w_slots`; `PY_NULL` means that no slot has
    /// been assigned yet.
    pub w_slots: PyObjectRef,
}

/// GC type id assigned to `W_ListObject` at `JitDriver` init time.
/// Held as a constant here (rather than runtime-queried) so
/// pyre-object's host-side allocator can reach it without a
/// back-channel; `pyre-jit/src/eval.rs` asserts the same id is
/// returned by `gc.register_type(...)` so any drift panics on
/// startup. Re-exported from `pyre_jit_trace::descr` for existing
/// call sites.
pub const W_LIST_GC_TYPE_ID: u32 = 7;
pub const W_LIST_OBJECT_SIZE: usize = std::mem::size_of::<W_ListObject>();

/// Allocate the translated `SizeListStrategy` instance payload.  The object
/// space reference is process-global in pyre, leaving its sole per-instance
/// field (`sizehint`) as one Signed cell.  This is GC state, not a Python int:
/// diagnostic APIs must not expose the strategy implementation as a W_Root.
unsafe fn alloc_sizehint_state(sizehint: i64) -> *mut TypedItemsBlock {
    let state = alloc_typed_items_block(1, gc_int_array_gc_type_id());
    *(typed_items_block_items_base(state) as *mut i64) = sizehint;
    state
}

#[inline]
unsafe fn sizehint_state_value(state: *mut ItemsBlock) -> i64 {
    *(typed_items_block_items_base(state as *mut TypedItemsBlock) as *const i64)
}

#[inline]
unsafe fn set_sizehint_state_value(state: *mut ItemsBlock, sizehint: i64) {
    *(typed_items_block_items_base(state as *mut TypedItemsBlock) as *mut i64) = sizehint;
}

/// Allocate the immutable erased tuple used by PyPy's range list strategies.
/// `SimpleRangeListStrategy` stores `(length,)`; `RangeListStrategy` stores
/// `(start, step, length)`.  These are RPython Signed cells, not Python ints.
unsafe fn alloc_range_state(values: &[i64]) -> *mut TypedItemsBlock {
    let state = alloc_typed_items_block(values.len(), gc_int_array_gc_type_id());
    std::ptr::copy_nonoverlapping(
        values.as_ptr(),
        typed_items_block_items_base(state) as *mut i64,
        values.len(),
    );
    state
}

#[inline]
unsafe fn range_state_value(state: *mut ItemsBlock, index: usize) -> i64 {
    *((typed_items_block_items_base(state as *mut TypedItemsBlock) as *const i64).add(index))
}

#[inline]
unsafe fn range_list_length(list: &W_ListObject) -> usize {
    let index = if list.strategy == ListStrategy::SimpleRange {
        0
    } else {
        2
    };
    usize::try_from(range_state_value(list.items, index)).unwrap()
}

#[inline]
unsafe fn range_list_start_step(list: &W_ListObject) -> (i64, i64) {
    if list.strategy == ListStrategy::SimpleRange {
        (0, 1)
    } else {
        (
            range_state_value(list.items, 0),
            range_state_value(list.items, 1),
        )
    }
}

#[inline]
unsafe fn range_list_item_unchecked(list: &W_ListObject, index: usize) -> i64 {
    let (start, step) = range_list_start_step(list);
    start + (index as i64) * step
}

unsafe fn range_list_values(list: &W_ListObject) -> Vec<i64> {
    let length = range_list_length(list);
    (0..length)
        .map(|index| range_list_item_unchecked(list, index))
        .collect()
}

impl W_ListObject {
    /// `_Py_atomic_load_ssize_relaxed(&ob_size)` on the Object-strategy length.
    ///
    /// Not the list's length — that is `w_list_len`, which dispatches on the
    /// strategy.  This is the raw slot, valid only where the caller has already
    /// established the Object strategy.
    #[inline]
    pub fn length_relaxed(&self) -> usize {
        self.length.load(Ordering::Relaxed)
    }

    /// `_Py_atomic_store_ssize_relaxed(&ob_size, n)` on the same slot.
    #[inline]
    pub fn set_length_relaxed(&self, n: usize) {
        self.length.store(n, Ordering::Relaxed);
    }

    #[inline]
    fn live_len(&self) -> usize {
        match self.strategy {
            ListStrategy::Empty | ListStrategy::Size => 0,
            ListStrategy::SimpleRange | ListStrategy::Range => unsafe { range_list_length(self) },
            ListStrategy::Object => self.length_relaxed(),
            // rlist.py `ll_length` is the `list.int_len` / `list.float_len`
            // oopspec leaf.  Keep that call boundary: jtransform lowers the
            // nested `int_items.len` / `float_items.len` path to one field read
            // off the list owner.  Inlining the Rust field path instead first
            // materialises the by-value storage struct as a GC Ref and then
            // reads through it, which is neither an RPython getsubstruct nor a
            // valid pointer.
            ListStrategy::Integer | ListStrategy::IntOrFloat => ll_list_int_length(self),
            ListStrategy::Float => ll_list_float_length(self),
            ListStrategy::Bytes => self.bytes_items.len,
            ListStrategy::Ascii => self.ascii_items.len,
        }
    }

    /// CPython 3.14 `list_resize`'s `allocated` calculation.
    fn resized_allocation(&self, old_size: usize, new_size: usize) -> usize {
        let allocated = if self.allocated < 0 {
            0
        } else {
            self.allocated as usize
        };
        if allocated >= new_size && new_size >= (allocated >> 1) {
            return allocated;
        }
        let mut new_allocated = (new_size + (new_size >> 3) + 6) & !3;
        if new_size > old_size && new_size - old_size > new_allocated.saturating_sub(new_size) {
            new_allocated = (new_size + 3) & !3;
        }
        if new_size == 0 {
            new_allocated = 0;
        }
        new_allocated
    }

    fn sync_allocated(&mut self, old_size: usize) {
        self.allocated = self.resized_allocation(old_size, self.live_len()) as isize;
    }

    #[inline]
    unsafe fn object_items_capacity(&self) -> usize {
        items_block_capacity(self.items)
    }

    #[inline]
    unsafe fn object_spare_capacity(&self) -> usize {
        self.object_items_capacity()
            .saturating_sub(self.length_relaxed())
    }

    /// Grow `items` to accommodate at least `min_cap` slots. Upstream
    /// `_ll_list_resize_really` (rlist.py) — allocate fresh,
    /// copy, swap, free.
    unsafe fn object_grow(obj: PyObjectRef, min_cap: usize) -> PyObjectRef {
        let _roots = crate::gc_roots::push_roots();
        let obj_slot = crate::gc_roots::shadow_stack_len();
        let obj = crate::gc_roots::pin_root(obj);
        let extra = if min_cap < 9 { 3 } else { 6 };
        let target_cap = min_cap.saturating_add(extra).saturating_add(min_cap >> 3);
        // The GC rewrite emits COND_CALL_GC_WB before SETFIELD_GC. Keep that
        // ordering on the host path too: the grow below allocates in the moving
        // nursery and may collect, so this old list has to be on the remembered
        // set before that minor runs, not after it.
        list_write_barrier(obj);
        // Phase L2: a GC-managed grow allocates the new block in the moving
        // nursery and may collect; `grow_list_items_block_gc` roots the old
        // block's live items across that allocation. Callers that hold an
        // incoming `value` across this call root it themselves before grow.
        let new_items_slot = crate::gc_roots::shadow_stack_len();
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        let list = &mut *(obj as *mut W_ListObject);
        let new_items = grow_list_items_block_gc(list.items, target_cap, list.length_relaxed());
        let _ = crate::gc_roots::pin_root(new_items as PyObjectRef);
        // framework.py's GC transform places the owner write barrier directly
        // before SETFIELD_GC. Keep the old block installed while the barrier
        // runs, then swap: remembering an old object that does not hold the new
        // young pointer yet is the harmless direction, and it is the one that
        // survives a collection point ever appearing between the two.
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        list_write_barrier(obj);
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        let list = &mut *(obj as *mut W_ListObject);
        let old_items = list.items;
        list.items = crate::gc_roots::shadow_stack_get(new_items_slot) as *mut ItemsBlock;
        dealloc_list_items_block(old_items);
        crate::gc_roots::shadow_stack_get(obj_slot)
    }

    /// `AbstractUnwrappedStrategy.get_empty_storage(sizehint)` for the Object
    /// strategy: publish an exact-capacity, zero-length RPython list backing.
    unsafe fn object_resize_capacity(obj: PyObjectRef, capacity: usize) -> PyObjectRef {
        Self::try_object_resize_capacity(obj, capacity)
            .unwrap_or_else(|| crate::object_array::items_block_alloc_failed(capacity))
    }

    /// [`W_ListObject::object_resize_capacity`] for a capacity that came from
    /// Python: the fresh block is the only fallible step and it precedes every
    /// store, so a refusal leaves the list on the block it already had.
    unsafe fn try_object_resize_capacity(obj: PyObjectRef, capacity: usize) -> Option<PyObjectRef> {
        let _roots = crate::gc_roots::push_roots();
        let obj_slot = crate::gc_roots::shadow_stack_len();
        let obj = crate::gc_roots::pin_root(obj);
        let list = &*(obj as *const W_ListObject);
        assert!(capacity >= list.length_relaxed());
        if capacity == list.object_items_capacity() {
            return Some(obj);
        }
        if capacity == 0 {
            list_write_barrier(obj);
            let obj = crate::gc_roots::shadow_stack_get(obj_slot);
            let list = &mut *(obj as *mut W_ListObject);
            let old = list.items;
            list.items = std::ptr::null_mut();
            dealloc_list_items_block(old);
            return Some(crate::gc_roots::shadow_stack_get(obj_slot));
        }
        list_write_barrier(obj);
        let block_slot = crate::gc_roots::shadow_stack_len();
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        let list = &*(obj as *const W_ListObject);
        // A refusal here leaves the barrier above spent on a list that keeps
        // its old block: one extra remembered-set entry, and nothing else.
        let block = crate::object_array::try_grow_list_items_block_gc(
            list.items,
            capacity,
            list.length_relaxed(),
        )?;
        let _ = crate::gc_roots::pin_root(block as PyObjectRef);
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        list_write_barrier(obj);
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        let list = &mut *(obj as *mut W_ListObject);
        let old = list.items;
        list.items = crate::gc_roots::shadow_stack_get(block_slot) as *mut ItemsBlock;
        dealloc_list_items_block(old);
        Some(crate::gc_roots::shadow_stack_get(obj_slot))
    }

    /// Grow `bytes_items` to accommodate at least `min_cap` slots — the
    /// Bytes-strategy counterpart of [`W_ListObject::object_grow`], and the
    /// only place a fresh `bytes_items` block may be published.
    ///
    /// The grow has to be driven from the list. `BytesArray` cannot reach its
    /// owner, so growing from inside it allocates the block and stores it into
    /// the list as one step, with no way to barrier the list in between. A
    /// barrier the caller ran before the call is spent by the collection that
    /// allocation itself starts: the list leaves the remembered set again, and
    /// the young block then reaches an old list that the next minor collection
    /// never visits, which drops the block and every `BytesBlock` reachable
    /// only through it. Barrier on both sides of the allocation, with the fresh
    /// block rooted across the second one, exactly as `object_grow` does.
    unsafe fn bytes_grow(obj: PyObjectRef, min_cap: usize) -> PyObjectRef {
        let _roots = crate::gc_roots::push_roots();
        let obj_slot = crate::gc_roots::shadow_stack_len();
        let obj = crate::gc_roots::pin_root(obj);
        let extra = if min_cap < 9 { 3 } else { 6 };
        let target_cap = min_cap.saturating_add(extra).saturating_add(min_cap >> 3);
        list_write_barrier(obj);
        let new_block_slot = crate::gc_roots::shadow_stack_len();
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        let list = &*(obj as *const W_ListObject);
        let new_block =
            grow_list_items_block_gc(list.bytes_items.block, target_cap, list.bytes_items.len());
        let _ = crate::gc_roots::pin_root(new_block as PyObjectRef);
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        list_write_barrier(obj);
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        let list = &mut *(obj as *mut W_ListObject);
        let old_block = list.bytes_items.block;
        list.bytes_items.block =
            crate::gc_roots::shadow_stack_get(new_block_slot) as *mut ItemsBlock;
        dealloc_list_items_block(old_block);
        crate::gc_roots::shadow_stack_get(obj_slot)
    }

    /// RPython `_ll_list_resize_hint_really` for BytesListStrategy.
    unsafe fn bytes_resize_capacity(obj: PyObjectRef, capacity: usize) -> PyObjectRef {
        Self::try_bytes_resize_capacity(obj, capacity)
            .unwrap_or_else(|| crate::object_array::items_block_alloc_failed(capacity))
    }

    /// [`W_ListObject::bytes_resize_capacity`] for a capacity that came from
    /// Python: the fresh block is the only fallible step and it precedes every
    /// store, so a refusal leaves the list on the block it already had.
    unsafe fn try_bytes_resize_capacity(obj: PyObjectRef, capacity: usize) -> Option<PyObjectRef> {
        let _roots = crate::gc_roots::push_roots();
        let obj_slot = crate::gc_roots::shadow_stack_len();
        let obj = crate::gc_roots::pin_root(obj);
        let list = &*(obj as *const W_ListObject);
        assert!(capacity >= list.bytes_items.len());
        if capacity == list.bytes_items.heap_capacity() {
            return Some(obj);
        }
        if capacity == 0 {
            let list = &mut *(obj as *mut W_ListObject);
            let old = list.bytes_items.block;
            list.bytes_items.block = std::ptr::null_mut();
            list.bytes_items.set_len(0);
            dealloc_list_items_block(old);
            return Some(crate::gc_roots::shadow_stack_get(obj_slot));
        }
        list_write_barrier(obj);
        let block_slot = crate::gc_roots::shadow_stack_len();
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        let list = &*(obj as *const W_ListObject);
        // A refusal here leaves the barrier above spent on a list that keeps
        // its old block: one extra remembered-set entry, and nothing else.
        let block = crate::object_array::try_grow_list_items_block_gc(
            list.bytes_items.block,
            capacity,
            list.bytes_items.len(),
        )?;
        let _ = crate::gc_roots::pin_root(block as PyObjectRef);
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        list_write_barrier(obj);
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        let list = &mut *(obj as *mut W_ListObject);
        let old = list.bytes_items.block;
        list.bytes_items.block = crate::gc_roots::shadow_stack_get(block_slot) as *mut ItemsBlock;
        dealloc_list_items_block(old);
        Some(crate::gc_roots::shadow_stack_get(obj_slot))
    }

    /// Publish an already-built `BytesArray` as this list's `bytes_items`,
    /// under [`W_ListObject::bytes_grow`]'s barrier discipline.
    ///
    /// `fresh` holds a block that nothing roots yet, so it travels on the
    /// shadow stack across the owner barrier — the barrier waits on the GC
    /// operation gate and can therefore let a collection move the block before
    /// `install` pins it.
    unsafe fn install_bytes_items(obj: PyObjectRef, fresh: BytesArray) -> PyObjectRef {
        let _roots = crate::gc_roots::push_roots();
        let obj_slot = crate::gc_roots::shadow_stack_len();
        let obj = crate::gc_roots::pin_root(obj);
        let block_slot = fresh.pin_block();
        list_write_barrier(obj);
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        let list = &mut *(obj as *mut W_ListObject);
        let mut fresh = fresh;
        fresh.reload_block(block_slot);
        list.bytes_items.install(fresh);
        crate::gc_roots::shadow_stack_get(obj_slot)
    }

    /// AsciiListStrategy counterpart of [`W_ListObject::bytes_grow`].
    unsafe fn ascii_grow(obj: PyObjectRef, min_cap: usize) -> PyObjectRef {
        let _roots = crate::gc_roots::push_roots();
        let obj_slot = crate::gc_roots::shadow_stack_len();
        let obj = crate::gc_roots::pin_root(obj);
        let extra = if min_cap < 9 { 3 } else { 6 };
        let target_cap = min_cap.saturating_add(extra).saturating_add(min_cap >> 3);
        list_write_barrier(obj);
        let new_block_slot = crate::gc_roots::shadow_stack_len();
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        let list = &*(obj as *const W_ListObject);
        let new_block =
            grow_list_items_block_gc(list.ascii_items.block, target_cap, list.ascii_items.len());
        let _ = crate::gc_roots::pin_root(new_block as PyObjectRef);
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        list_write_barrier(obj);
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        let list = &mut *(obj as *mut W_ListObject);
        let old_block = list.ascii_items.block;
        list.ascii_items.block =
            crate::gc_roots::shadow_stack_get(new_block_slot) as *mut ItemsBlock;
        dealloc_list_items_block(old_block);
        crate::gc_roots::shadow_stack_get(obj_slot)
    }

    /// RPython `_ll_list_resize_hint_really` for AsciiListStrategy.
    unsafe fn ascii_resize_capacity(obj: PyObjectRef, capacity: usize) -> PyObjectRef {
        Self::try_ascii_resize_capacity(obj, capacity)
            .unwrap_or_else(|| crate::object_array::items_block_alloc_failed(capacity))
    }

    /// [`W_ListObject::ascii_resize_capacity`] for a capacity that came from
    /// Python: the fresh block is the only fallible step and it precedes every
    /// store, so a refusal leaves the list on the block it already had.
    unsafe fn try_ascii_resize_capacity(obj: PyObjectRef, capacity: usize) -> Option<PyObjectRef> {
        let _roots = crate::gc_roots::push_roots();
        let obj_slot = crate::gc_roots::shadow_stack_len();
        let obj = crate::gc_roots::pin_root(obj);
        let list = &*(obj as *const W_ListObject);
        assert!(capacity >= list.ascii_items.len());
        if capacity == list.ascii_items.heap_capacity() {
            return Some(obj);
        }
        if capacity == 0 {
            let list = &mut *(obj as *mut W_ListObject);
            let old = list.ascii_items.block;
            list.ascii_items.block = std::ptr::null_mut();
            list.ascii_items.set_len(0);
            dealloc_list_items_block(old);
            return Some(crate::gc_roots::shadow_stack_get(obj_slot));
        }
        list_write_barrier(obj);
        let block_slot = crate::gc_roots::shadow_stack_len();
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        let list = &*(obj as *const W_ListObject);
        // A refusal here leaves the barrier above spent on a list that keeps
        // its old block: one extra remembered-set entry, and nothing else.
        let block = crate::object_array::try_grow_list_items_block_gc(
            list.ascii_items.block,
            capacity,
            list.ascii_items.len(),
        )?;
        let _ = crate::gc_roots::pin_root(block as PyObjectRef);
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        list_write_barrier(obj);
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        let list = &mut *(obj as *mut W_ListObject);
        let old = list.ascii_items.block;
        list.ascii_items.block = crate::gc_roots::shadow_stack_get(block_slot) as *mut ItemsBlock;
        dealloc_list_items_block(old);
        Some(crate::gc_roots::shadow_stack_get(obj_slot))
    }

    /// Publish an already-built AsciiListStrategy backing array with the
    /// owner barrier directly before its field store.
    unsafe fn install_ascii_items(obj: PyObjectRef, fresh: UnicodeArray) -> PyObjectRef {
        let _roots = crate::gc_roots::push_roots();
        let obj_slot = crate::gc_roots::shadow_stack_len();
        let obj = crate::gc_roots::pin_root(obj);
        let block_slot = fresh.pin_block();
        list_write_barrier(obj);
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        let list = &mut *(obj as *mut W_ListObject);
        let mut fresh = fresh;
        fresh.reload_block(block_slot);
        list.ascii_items.install(fresh);
        crate::gc_roots::shadow_stack_get(obj_slot)
    }

    /// Upstream list.append equivalent for the object strategy.
    /// (listobject.py `AbstractUnwrappedStrategy.append` for the
    /// Object case: no unwrap, just append.)
    /// `pub` because the blackhole calls it by address: the #171 fold descends
    /// `w_list_append`, so a guard exit inside that body resumes in its jitcode
    /// and re-executes this store as a `residual_call`, whose funcptr comes
    /// from the `jit_fnaddr.rs` binding (pyre's stand-in for `call.py:181-183
    /// getfunctionptr(graph)`).
    ///
    /// # Safety
    /// `self` must be an Object-strategy list.
    pub unsafe fn object_push(&mut self, value: PyObjectRef) {
        let _roots = crate::gc_roots::push_roots();
        let root_base = crate::gc_roots::shadow_stack_len();
        crate::gc_roots::publish_roots(&[self as *mut W_ListObject as PyObjectRef, value]);
        crate::gc_roots::normalize_roots(root_base, 2);
        let obj = crate::gc_roots::shadow_stack_get(root_base);
        let list = &mut *(obj as *mut W_ListObject);
        // At capacity, route the grow through the `dont_look_inside`
        // boundary: it roots `value` across the (collecting) resize and
        // returns it relocated. The in-place store below stays outside the
        // boundary so the spare-capacity fold still lowers it. No-op grow
        // under the std::alloc fallback.
        let value = if list.length_relaxed() == list.object_items_capacity() {
            w_list_grow_items_block(obj, crate::gc_roots::shadow_stack_get(root_base + 1))
        } else {
            crate::gc_roots::shadow_stack_get(root_base + 1)
        };
        let obj = crate::gc_roots::shadow_stack_get(root_base);
        let value = prepare_list_ref_store(obj, value);
        let obj = crate::gc_roots::shadow_stack_get(root_base);
        let list = &mut *(obj as *mut W_ListObject);
        let base = items_block_items_base(list.items);
        *base.add(list.length_relaxed()) = value;
        list.set_length_relaxed(list.length_relaxed() + 1);
    }

    unsafe fn object_insert(&mut self, index: usize, value: PyObjectRef) {
        let _roots = crate::gc_roots::push_roots();
        let root_base = crate::gc_roots::shadow_stack_len();
        crate::gc_roots::publish_roots(&[self as *mut W_ListObject as PyObjectRef, value]);
        crate::gc_roots::normalize_roots(root_base, 2);
        let obj = crate::gc_roots::shadow_stack_get(root_base);
        let list = &mut *(obj as *mut W_ListObject);
        assert!(index <= list.length_relaxed());
        // Same grow-then-store shape as `object_push`: at capacity, route the
        // grow through the `dont_look_inside` boundary, which roots `value`
        // across the (collecting) resize and returns it relocated.
        let value = if list.length_relaxed() == list.object_items_capacity() {
            w_list_grow_items_block(obj, crate::gc_roots::shadow_stack_get(root_base + 1))
        } else {
            crate::gc_roots::shadow_stack_get(root_base + 1)
        };
        let obj = crate::gc_roots::shadow_stack_get(root_base);
        let value = prepare_list_ref_store(obj, value);
        // The shift below moves items across card pages; generalize first.
        // The barrier answers with the list's post-barrier address, so this is
        // the reload the following store needs as well.
        let obj = list_before_move_barrier(crate::gc_roots::shadow_stack_get(root_base));
        let list = &mut *(obj as *mut W_ListObject);
        let base = items_block_items_base(list.items);
        let p = base.add(index);
        std::ptr::copy(p, p.add(1), list.length_relaxed() - index);
        *p = value;
        list.set_length_relaxed(list.length_relaxed() + 1);
    }

    unsafe fn object_remove(&mut self, index: usize) -> PyObjectRef {
        assert!(index < self.length_relaxed());
        // The barrier's ownership query is a safepoint, so `self` is reloaded
        // through the address it answers with rather than reused.
        let obj = list_before_move_barrier(self as *mut W_ListObject as PyObjectRef);
        let this = &mut *(obj as *mut W_ListObject);
        let base = items_block_items_base(this.items);
        let value = *base.add(index);
        let p = base.add(index);
        std::ptr::copy(p.add(1), p, this.length_relaxed() - index - 1);
        // Phase L2: the varsize walker forwards items[0..capacity], so clear the
        // vacated tail slot the shift left holding a stale duplicate.
        *base.add(this.length_relaxed() - 1) = PY_NULL;
        this.set_length_relaxed(this.length_relaxed() - 1);
        value
    }

    unsafe fn object_reverse(&mut self) {
        // A permutation moves pointers across card pages without storing any
        // new reference, so it owes the same barrier as the shifts above.
        let obj = list_before_move_barrier(self as *mut W_ListObject as PyObjectRef);
        let this = &mut *(obj as *mut W_ListObject);
        // `rlist.py`'s `ll_reverse` operates on the logical list directly
        // through ll_getitem_fast / ll_setitem_fast. Keep that shape here
        // instead of manufacturing a Rust fat slice over the over-allocated
        // items block.
        //
        // Its `@jit.look_inside_iff(lambda l: jit.isvirtual(l) and
        // jit.isconstant(l.ll_length()))` is not carried: pyre has no
        // `jit.isvirtual` / `jit.isconstant` intrinsic, so the predicate
        // cannot be spelled — the same gap the four `look_inside_iff` notes
        // in `dictmultiobject.rs` record.
        let base = items_block_items_base(this.items);
        let mut i = 0;
        let mut length_1_i = this.length_relaxed() as isize - 1;
        while i < length_1_i {
            let tmp = *base.add(i as usize);
            *base.add(i as usize) = *base.add(length_1_i as usize);
            *base.add(length_1_i as usize) = tmp;
            i += 1;
            length_1_i -= 1;
        }
    }

    unsafe fn object_drain(&mut self, range: std::ops::Range<usize>) {
        let start = range.start;
        let end = range.end;
        assert!(start <= end && end <= self.length_relaxed());
        let count = end - start;
        if count == 0 {
            return;
        }
        // The barrier's ownership query is a safepoint, so `self` is reloaded
        // through the address it answers with rather than reused.
        let obj = list_before_move_barrier(self as *mut W_ListObject as PyObjectRef);
        let this = &mut *(obj as *mut W_ListObject);
        let base = items_block_items_base(this.items);
        let p = base.add(start);
        std::ptr::copy(p.add(count), p, this.length_relaxed() - end);
        let old_len = this.length_relaxed();
        this.set_length_relaxed(this.length_relaxed() - count);
        // Phase L2: clear the vacated tail [new_len..old_len] the shift left
        // holding stale duplicates, so the varsize walker (0..capacity) skips them.
        for i in this.length_relaxed()..old_len {
            *base.add(i) = PY_NULL;
        }
    }

    unsafe fn object_splice(
        &mut self,
        start: usize,
        remove_count: usize,
        new_values: &[PyObjectRef],
    ) {
        let old_len = self.length_relaxed();
        let s = start.min(old_len);
        let slicelength = remove_count.min(old_len - s);
        let len2 = new_values.len();
        let new_len = old_len - slicelength + len2;
        // Root the incoming values across a possible (collecting) grow, then
        // write them from the relocated slots. No-op under the std::alloc
        // fallback; covers both the grow and no-grow branches uniformly.
        let _roots = crate::gc_roots::push_roots();
        let obj_slot = crate::gc_roots::shadow_stack_len();
        crate::gc_roots::publish_roots(&[self as *mut W_ListObject as PyObjectRef]);
        let save = crate::gc_roots::shadow_stack_len();
        crate::gc_roots::publish_roots(new_values);
        crate::gc_roots::normalize_roots(obj_slot, new_values.len() + 1);
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        list_write_barrier(obj);
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        let list = &mut *(obj as *mut W_ListObject);
        if len2 > slicelength {
            if new_len > list.object_items_capacity() {
                let obj = W_ListObject::object_grow(obj, new_len);
                let list = &mut *(obj as *mut W_ListObject);
                let base = items_block_items_base(list.items);
                std::ptr::copy(
                    base.add(s + slicelength),
                    base.add(s + len2),
                    old_len - s - slicelength,
                );
                list.set_length_relaxed(new_len);
            } else {
                let base = items_block_items_base(list.items);
                std::ptr::copy(
                    base.add(s + slicelength),
                    base.add(s + len2),
                    old_len - s - slicelength,
                );
                list.set_length_relaxed(new_len);
            }
        } else if slicelength > len2 {
            let base = items_block_items_base(list.items);
            std::ptr::copy(
                base.add(s + slicelength),
                base.add(s + len2),
                old_len - s - slicelength,
            );
            // Shrinking splice: clear the vacated tail [new_len..old_len].
            for i in new_len..old_len {
                *base.add(i) = PY_NULL;
            }
            list.set_length_relaxed(new_len);
        }
        if len2 > 0 {
            let obj = crate::gc_roots::shadow_stack_get(obj_slot);
            let list = &mut *(obj as *mut W_ListObject);
            let base = items_block_items_base(list.items);
            for i in 0..len2 {
                *base.add(s + i) = crate::gc_roots::shadow_stack_get(save + i);
            }
        }
    }

    unsafe fn object_to_vec(&self) -> Vec<PyObjectRef> {
        let base = items_block_items_base(self.items);
        let length = self.length_relaxed();
        let mut result = Vec::with_capacity(length);
        for i in 0..length {
            result.push(*base.add(i));
        }
        result
    }

    /// Free the current `items` block and install a freshly allocated
    /// one populated with `values`. `length` is reset to `values.len()`.
    unsafe fn set_object_items_from_vec(&mut self, values: Vec<PyObjectRef>) {
        let _roots = crate::gc_roots::push_roots();
        let root_base = crate::gc_roots::shadow_stack_len();
        crate::gc_roots::publish_roots(&[self as *mut W_ListObject as PyObjectRef]);
        crate::gc_roots::publish_roots(&values);
        crate::gc_roots::normalize_roots(root_base, values.len() + 1);
        let obj = crate::gc_roots::shadow_stack_get(root_base);
        list_write_barrier(obj);
        // The pre-store barrier may wait behind a moving collection.  The
        // shadow slots are authoritative afterwards; the input Vec still
        // contains the pre-collection copies.
        let mut rooted_values = Vec::with_capacity(values.len());
        for i in 0..values.len() {
            rooted_values.push(crate::gc_roots::shadow_stack_get(root_base + 1 + i));
        }
        let new_items_slot = crate::gc_roots::shadow_stack_len();
        let new_items = alloc_list_items_block_gc(&rooted_values);
        let _ = crate::gc_roots::pin_root(new_items as PyObjectRef);
        // `_ll_list_resize_really` installs the freshly allocated GcArray
        // through SETFIELD_GC.  Run the owner barrier before publishing the
        // edge, the same direction the transform emits: an old object on the
        // remembered set without the young pointer yet is harmless, an old
        // object holding it without being remembered is not.
        let obj = crate::gc_roots::shadow_stack_get(root_base);
        list_write_barrier(obj);
        let obj = crate::gc_roots::shadow_stack_get(root_base);
        let list = &mut *(obj as *mut W_ListObject);
        let old_items = list.items;
        list.items = crate::gc_roots::shadow_stack_get(new_items_slot) as *mut ItemsBlock;
        list.set_length_relaxed(rooted_values.len());
        dealloc_list_items_block(old_items);
    }

    /// Drop the object-strategy backing (used when switching to a typed
    /// strategy). Sets `items = null` and `length = 0`.
    unsafe fn drop_object_items(&mut self) {
        dealloc_list_items_block(self.items);
        self.items = std::ptr::null_mut();
        self.set_length_relaxed(0);
    }
}

/// `interp_magic.py strategy` — concrete list strategy class name.
///
/// # Safety
/// `obj` must point to a live `W_ListObject`.
#[inline]
pub unsafe fn w_list_strategy_name(obj: PyObjectRef) -> &'static str {
    unsafe { (*(obj as *const W_ListObject)).strategy.class_name() }
}

/// Grow the Object-strategy backing of `obj` to hold at least one more
/// element and return `value` relocated to its post-collection address.
///
/// Residualized: the grow drives the moving collector through `object_grow`
/// → `grow_list_items_block_gc`'s `push_roots` / `pin_root` /
/// `shadow_stack_get` / `alloc_items_block_gc` — shadow-stack and moving-GC
/// plumbing the tracer cannot model, the same reason `w_list_new`
/// residualizes. The JIT leaves the call as a residual returning the
/// relocated value pointer rather than tracing into the resize allocator.
/// The in-place store (`items[len] = value; length += 1`) stays with the
/// caller, outside this boundary, so the spare-capacity fold still lowers
/// it to `setarrayitem` + `set_len`.
///
/// `_ll_list_resize_ge`'s realloc case (rlist.py): `value` is pinned
/// across the (collecting) grow and read back from its relocated
/// shadow-stack slot — `grow_list_items_block_gc` may move it during its
/// collection, so the returned pointer, not the stale argument, is what the
/// caller must store.
///
/// # Safety
/// `obj` must point to a valid Object-strategy `W_ListObject`; `value` must
/// be a live `PyObjectRef`.
#[majit_macros::dont_look_inside]
pub unsafe fn w_list_grow_items_block(obj: PyObjectRef, value: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let save = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    let _ = crate::gc_roots::pin_root(value);
    let obj = crate::gc_roots::shadow_stack_get(save);
    let list = &mut *(obj as *mut W_ListObject);
    W_ListObject::object_grow(obj, list.length_relaxed() + 1);
    crate::gc_roots::shadow_stack_get(save + 1)
}

/// [`w_list_grow_items_block`] for the Bytes strategy: makes room for one more
/// erased `rpython str` and returns `value` at its post-grow address.
///
/// `value` is pinned across the grow for the same reason the object arm pins
/// it — [`W_ListObject::bytes_grow`] allocates in the moving nursery and may
/// collect, so the caller must store the returned pointer, not the argument it
/// passed.
///
/// # Safety
/// `obj` must point to a valid Bytes-strategy `W_ListObject`; `value` must be
/// a live `PyObjectRef`.
#[majit_macros::dont_look_inside]
pub unsafe fn w_list_grow_bytes_block(obj: PyObjectRef, value: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let save = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    let _ = crate::gc_roots::pin_root(value);
    let obj = crate::gc_roots::shadow_stack_get(save);
    let list = &*(obj as *const W_ListObject);
    W_ListObject::bytes_grow(obj, list.bytes_items.len() + 1);
    crate::gc_roots::shadow_stack_get(save + 1)
}

/// [`w_list_grow_items_block`] for AsciiListStrategy's erased UTF-8 storage.
#[majit_macros::dont_look_inside]
pub unsafe fn w_list_grow_ascii_block(obj: PyObjectRef, value: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let save = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    let _ = crate::gc_roots::pin_root(value);
    let obj = crate::gc_roots::shadow_stack_get(save);
    let list = &*(obj as *const W_ListObject);
    W_ListObject::ascii_grow(obj, list.ascii_items.len() + 1);
    crate::gc_roots::shadow_stack_get(save + 1)
}

/// listobject.py is_plain_int1(w_obj)
///
/// Accepts exact W_IntObject (not bool, not int subclass) or W_LongObject
/// whose value fits in a machine-word integer. Shared with
/// `specialisedtupleobject.py makespecialisedtuple2` and the
/// `IntegerListStrategy.is_correct_type` strategy gate
/// (`listobject.py:1957-1958`).
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_plain_int1(item: PyObjectRef) -> bool {
    if item.is_null() {
        return false;
    }
    // A tagged immediate is always an exact `int` (never a subclass and
    // never a bool), so it is a plain int without the `w_class` deref
    // below. Gated on `CAN_BE_TAGGED` (default false).
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(item) {
        return true;
    }
    // listobject.py `is_plain_int1`: `type(w_obj) is W_IntObject or
    // (type(w_obj) is W_LongObject and w_obj._fits_int())`. Layout identity
    // first (`is_int` is W_IntObject, `is_long` is W_LongObject); they do
    // not overlap. A non-fitting bigint stays off IntegerListStrategy.
    if is_int(item) && !is_bool(item) {
        // type(w_obj) is W_IntObject — reject int subclasses.
        // Subclass instances share ob_type == &INT_TYPE but have w_class
        // overwritten to the subclass type object (typedef.rs).
        let int_typeobj = get_instantiate(&INT_TYPE);
        let w_class = (*item).w_class;
        if int_typeobj.is_null() {
            return w_class.is_null();
        }
        if !w_class.is_null() && !std::ptr::eq(w_class, int_typeobj) {
            return false;
        }
        return true;
    }
    if is_long(item) {
        let int_typeobj = get_instantiate(&INT_TYPE);
        let w_class = (*item).w_class;
        if int_typeobj.is_null() {
            return w_class.is_null() && w_long_fits_int(item);
        }
        if !w_class.is_null() && !std::ptr::eq(w_class, int_typeobj) {
            return false;
        }
        return w_long_fits_int(item);
    }
    false
}

/// listobject.py `plain_int_w(space, w_obj)`. Unwraps a plain
/// int value from W_IntObject or W_LongObject. Caller must ensure
/// `is_plain_int1(item)` returned true (which for `W_LongObject`
/// implies `_fits_int()`). RPython routes through `w_obj._int_w(space)`
/// (`longobject.py:157`) which raises `OverflowError` on out-of-range
/// values; pyre treats that path as unreachable and panics on
/// precondition violation rather than silently returning 0.
#[inline]
pub(crate) unsafe fn plain_int_w(item: PyObjectRef) -> i64 {
    if is_int(item) {
        w_int_get_value(item)
    } else {
        // `is_plain_int1` has already performed the upstream `_fits_int`
        // guard.  Keep the same guard→`toint` split used by
        // `W_LongObject._int_w`; in particular, do not expose Rust's
        // `Result<i64, RBigIntError>` carrier to the translated graph.
        jit_bigint_to_i64_value(w_long_get_value(item))
    }
}

/// Check if all items are plain ints for IntegerListStrategy.
fn all_ints(items: &[PyObjectRef]) -> bool {
    items.iter().all(|&item| unsafe { is_plain_int1(item) })
}

/// Whether an item may enter PyPy's unboxed FloatListStrategy.
///
/// PyPy accepts every exact float because `W_FloatObject.is_w` treats equal
/// bit patterns as identical.  Python 3.14 instead uses pointer identity:
/// containers must retain the original NaN object so `[nan] == [nan]` is true
/// for one shared object while two freshly-created NaNs still compare false.
/// Keep NaNs in Object storage; ordinary exact floats retain PyPy's strategy.
///
/// Shared with JIT list stores so traced and concrete strategies agree.
///
/// # Safety
/// `item` must be null or point to a live object.
#[inline]
pub unsafe fn is_float_strategy_item(item: PyObjectRef) -> bool {
    !item.is_null() && is_plain_float_strict(item) && !w_float_get_value(item).is_nan()
}

/// Check if all items can use FloatListStrategy.
fn all_floats(items: &[PyObjectRef]) -> bool {
    items
        .iter()
        .all(|&item| unsafe { is_float_strategy_item(item) })
}

// rpython/rlib/longlong2float.py:90-150.  Keep these bit operations local to
// IntOrFloatListStrategy: the signed-longlong storage representation is part
// of the upstream strategy, not a general numeric coercion.
const INT_OR_FLOAT_INT_HIGH_WORD: u32 = 0xffff_fffe;

#[inline]
fn int_or_float_is_int(value: i64) -> bool {
    ((value as u64) >> 32) as u32 == INT_OR_FLOAT_INT_HIGH_WORD
}

#[inline]
fn int_or_float_encode_int(value: i64) -> Option<i64> {
    let value = i32::try_from(value).ok()?;
    Some(((INT_OR_FLOAT_INT_HIGH_WORD as u64) << 32 | value as u32 as u64) as i64)
}

#[inline]
fn int_or_float_encode_float(value: f64) -> Option<i64> {
    let bits = value.to_bits();
    (((bits >> 32) as u32) != INT_OR_FLOAT_INT_HIGH_WORD).then_some(bits as i64)
}

#[inline]
fn int_or_float_decode_int(value: i64) -> i64 {
    value as u32 as i32 as i64
}

#[inline]
fn int_or_float_as_float(value: i64) -> f64 {
    if int_or_float_is_int(value) {
        int_or_float_decode_int(value) as f64
    } else {
        f64::from_bits(value as u64)
    }
}

#[inline]
unsafe fn int_or_float_encode_item(item: PyObjectRef) -> Option<i64> {
    if is_plain_int1(item) {
        int_or_float_encode_int(plain_int_w(item))
    } else if is_float_strategy_item(item) {
        int_or_float_encode_float(w_float_get_value(item))
    } else {
        None
    }
}

fn all_int_or_float(items: &[PyObjectRef]) -> bool {
    items
        .iter()
        .all(|&item| unsafe { int_or_float_encode_item(item).is_some() })
}

fn boxed_from_int_or_float(values: &[i64], we_are_jitted: bool) -> Vec<PyObjectRef> {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    let mut previous = None;
    for &value in values {
        // listobject.py AbstractUnwrappedStrategy.getitems_copy reuses the
        // previous wrapper when IntOrFloatListStrategy._quick_cmp succeeds.
        let item = if !we_are_jitted && previous == Some(value) {
            crate::gc_roots::shadow_stack_get(crate::gc_roots::shadow_stack_len() - 1)
        } else if int_or_float_is_int(value) {
            w_int_new(int_or_float_decode_int(value))
        } else {
            w_float_new(f64::from_bits(value as u64))
        };
        let _ = crate::gc_roots::pin_root(item);
        previous = Some(value);
    }
    let mut items = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        items.push(crate::gc_roots::shadow_stack_get(root_base + i));
    }
    items
}

/// listobject.py IntegerListStrategy.switch_to_int_or_float_strategy.
unsafe fn integer_to_int_or_float(list: &mut W_ListObject) -> bool {
    let source = list.int_items.as_slice();
    let mut values = Vec::with_capacity(source.len());
    for &value in source {
        let Some(encoded) = int_or_float_encode_int(value) else {
            return false;
        };
        values.push(encoded);
    }
    list.int_items.install(IntArray::from_vec(values));
    list.strategy = ListStrategy::IntOrFloat;
    true
}

/// listobject.py FloatListStrategy.switch_to_int_or_float_strategy.
unsafe fn float_to_int_or_float(list: &mut W_ListObject) -> bool {
    let source = list.float_items.as_slice();
    let mut values = Vec::with_capacity(source.len());
    for &value in source {
        let Some(encoded) = int_or_float_encode_float(value) else {
            return false;
        };
        values.push(encoded);
    }
    list.int_items.install(IntArray::from_vec(values));
    list.float_items.install(FloatArray::from_vec(Vec::new()));
    list.strategy = ListStrategy::IntOrFloat;
    true
}

/// listobject.py AbstractUnwrappedStrategy.getitems_copy.
///
/// The two hints on the same function -- `getitems_unroll =
/// jit.unroll_safe(...)` and `getitems_copy = jit.look_inside_iff(lambda
/// self, w_list: w_list._unrolling_heuristic())(...)` -- are absent;
/// `look_inside_iff` needs `jit.isvirtual`/`isconstant`, which pyre has no
/// intrinsic for yet.
fn boxed_from_ints(values: &[i64], we_are_jitted: bool) -> Vec<PyObjectRef> {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    let mut previous = None;
    for &value in values {
        // IntegerListStrategy._quick_cmp compares the unboxed values.
        let item = if !we_are_jitted && previous == Some(value) {
            crate::gc_roots::shadow_stack_get(crate::gc_roots::shadow_stack_len() - 1)
        } else {
            w_int_new(value)
        };
        let _ = crate::gc_roots::pin_root(item);
        previous = Some(value);
    }
    let mut items = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        items.push(crate::gc_roots::shadow_stack_get(root_base + i));
    }
    items
}

fn boxed_from_floats(values: &[f64], we_are_jitted: bool) -> Vec<PyObjectRef> {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    let mut previous_bits = None;
    for &value in values {
        // FloatListStrategy._quick_cmp compares the raw IEEE-754 payload.
        let bits = value.to_bits();
        let item = if !we_are_jitted && previous_bits == Some(bits) {
            crate::gc_roots::shadow_stack_get(crate::gc_roots::shadow_stack_len() - 1)
        } else {
            w_float_new(value)
        };
        let _ = crate::gc_roots::pin_root(item);
        previous_bits = Some(bits);
    }
    let mut items = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        items.push(crate::gc_roots::shadow_stack_get(root_base + i));
    }
    items
}

#[inline]
pub fn is_bytes_strategy_item(item: PyObjectRef) -> bool {
    unsafe { is_exact_type(item, &BYTES_TYPE) }
}

fn all_bytes(items: &[PyObjectRef]) -> bool {
    items.iter().all(|&item| is_bytes_strategy_item(item))
}

#[inline]
pub fn is_ascii_strategy_item(item: PyObjectRef) -> bool {
    unsafe { is_exact_type(item, &STR_TYPE) && w_str_is_ascii(item) }
}

fn all_ascii(items: &[PyObjectRef]) -> bool {
    items.iter().all(|&item| is_ascii_strategy_item(item))
}

/// Box each erased `rpython str` of the list pinned at `obj_slot`.
///
/// Unlike the int/float pair this cannot walk a slice taken once: every
/// `w_bytes_from_block` allocates, and a collection inside the loop forwards
/// `bytes_items.block`, so a base pointer captured up front goes on naming the
/// outgoing block.  Re-read the array from the pinned list at each step.
///
/// # Safety
/// `obj_slot` must hold a live `W_ListObject` in the Bytes strategy.
unsafe fn boxed_from_bytes(obj_slot: usize, we_are_jitted: bool) -> Vec<PyObjectRef> {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    let bytes_items = |slot: usize| -> &BytesArray {
        &(*(crate::gc_roots::shadow_stack_get(slot) as *const W_ListObject)).bytes_items
    };
    let len = bytes_items(obj_slot).len();
    for i in 0..len {
        // BytesListStrategy._quick_cmp is storage identity. Re-read both
        // slots through the rooted list because wrapping may move its block.
        let repeated = !we_are_jitted
            && i > 0
            && std::ptr::eq(
                bytes_items(obj_slot).as_slice()[i],
                bytes_items(obj_slot).as_slice()[i - 1],
            );
        let item = if repeated {
            crate::gc_roots::shadow_stack_get(crate::gc_roots::shadow_stack_len() - 1)
        } else {
            let value = bytes_items(obj_slot).as_slice()[i];
            w_bytes_from_block(value)
        };
        let _ = crate::gc_roots::pin_root(item);
    }
    (0..len)
        .map(|i| crate::gc_roots::shadow_stack_get(root_base + i))
        .collect()
}

/// Box each erased UTF-8 `rpython str` of the Ascii strategy, re-reading the
/// array through the rooted list after every allocating wrap.
unsafe fn boxed_from_ascii(obj_slot: usize, we_are_jitted: bool) -> Vec<PyObjectRef> {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    let ascii_items = |slot: usize| -> &UnicodeArray {
        &(*(crate::gc_roots::shadow_stack_get(slot) as *const W_ListObject)).ascii_items
    };
    let len = ascii_items(obj_slot).len();
    for i in 0..len {
        // AsciiListStrategy._quick_cmp is storage identity.
        let repeated = !we_are_jitted
            && i > 0
            && std::ptr::eq(
                ascii_items(obj_slot).as_slice()[i],
                ascii_items(obj_slot).as_slice()[i - 1],
            );
        let item = if repeated {
            crate::gc_roots::shadow_stack_get(crate::gc_roots::shadow_stack_len() - 1)
        } else {
            let value = ascii_items(obj_slot).as_slice()[i];
            w_str_from_storage(value as *mut _)
        };
        let _ = crate::gc_roots::pin_root(item);
    }
    (0..len)
        .map(|i| crate::gc_roots::shadow_stack_get(root_base + i))
        .collect()
}

/// Cold list strategy dehomogenization: a typed int/float list gained a
/// non-numeric element, so its unboxed backing storage is bulk re-boxed into
/// an Object-strategy items block one time.
///
/// `dont_look_inside`: the transition drives bulk backing-storage replacement;
/// `set_object_items_from_vec` / `IntArray::from_vec` / `FloatArray::from_vec`
/// tear down the typed storage through raw-Vec construction. Residualize the
/// whole cold transition via the registered fnaddr so the hot append/setitem
/// paths that call it stay traceable.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
#[majit_macros::dont_look_inside]
pub unsafe fn switch_to_object_strategy(list: &mut W_ListObject) -> PyObjectRef {
    if list.strategy == ListStrategy::Object {
        return list as *mut W_ListObject as PyObjectRef;
    }
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(list as *mut W_ListObject as PyObjectRef);
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let list = &mut *(obj as *mut W_ListObject);
    let seed: Vec<PyObjectRef> = match list.strategy {
        ListStrategy::SimpleRange | ListStrategy::Range => {
            let values = range_list_values(list);
            boxed_from_ints(&values, false)
        }
        ListStrategy::Integer => {
            let values = list.int_items.as_slice().to_vec();
            boxed_from_ints(&values, false)
        }
        ListStrategy::IntOrFloat => {
            let values = list.int_items.as_slice().to_vec();
            boxed_from_int_or_float(&values, false)
        }
        ListStrategy::Float => {
            let values = list.float_items.as_slice().to_vec();
            boxed_from_floats(&values, false)
        }
        ListStrategy::Bytes => boxed_from_bytes(obj_slot, false),
        ListStrategy::Ascii => boxed_from_ascii(obj_slot, false),
        ListStrategy::Object | ListStrategy::Empty | ListStrategy::Size => Vec::new(),
    };
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let list = &mut *(obj as *mut W_ListObject);
    if matches!(
        list.strategy,
        ListStrategy::Size | ListStrategy::SimpleRange | ListStrategy::Range
    ) {
        list.items = std::ptr::null_mut();
    }
    list.set_object_items_from_vec(seed);
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let list = &mut *(obj as *mut W_ListObject);
    // Take the strategy together with the block, ahead of the two `install`
    // calls below.  `list_object_custom_trace` reaches `items` only under the
    // Object strategy, and `IntArray::install` drops the outgoing storage
    // through `dealloc_typed_items_block`, whose ownership query is a
    // safepoint.  `set_object_items_from_vec` closed its own root scope on the
    // way out, so between its store and this line the fresh block's only
    // reference is the `items` edge the trace is still skipping — a collection
    // in that window reclaims it and leaves `items` naming reused nursery.
    list.strategy = ListStrategy::Object;
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let list = &mut *(obj as *mut W_ListObject);
    // Object strategy reads none of the typed arrays again, so drop all three
    // to the empty form instead of installing fresh single-slot blocks.  Each
    // `install` pins and reloads its incoming block, so it is a safepoint and
    // the list is re-read from its slot before the next one: writing a later
    // field through the reference the previous install left behind stores it
    // into the moved-from copy, and the live list keeps the outgoing block —
    // which the custom trace then forwards as a stale child.
    list.int_items.install(IntArray::empty());
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let list = &mut *(obj as *mut W_ListObject);
    list.float_items.install(FloatArray::empty());
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let list = &mut *(obj as *mut W_ListObject);
    list.bytes_items.install(BytesArray::empty());
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let list = &mut *(obj as *mut W_ListObject);
    list.ascii_items.install(UnicodeArray::empty());
    crate::gc_roots::shadow_stack_get(obj_slot)
}

/// `BaseRangeListStrategy.switch_to_integer_strategy`: materialise the
/// arithmetic progression into IntegerListStrategy exactly once before an
/// operation that destroys the range representation.
#[majit_macros::dont_look_inside]
unsafe fn switch_range_to_integer_strategy(list: &mut W_ListObject) -> PyObjectRef {
    debug_assert!(matches!(
        list.strategy,
        ListStrategy::SimpleRange | ListStrategy::Range
    ));
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(list as *mut W_ListObject as PyObjectRef);
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let values = range_list_values(&*(obj as *const W_ListObject));
    let fresh = IntArray::from_vec(values);
    let fresh_slot = fresh.pin_block();
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let list = &mut *(obj as *mut W_ListObject);
    list.int_items.install(fresh);
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let list = &mut *(obj as *mut W_ListObject);
    list.int_items.reload_block(fresh_slot);
    list.items = std::ptr::null_mut();
    list.strategy = ListStrategy::Integer;
    obj
}

/// Public `BaseRangeListStrategy.switch_to_integer_strategy` dispatch used by
/// interpreter-level operations whose generic body otherwise only reads the
/// list (slice, empty extend, and in-place repeat). Returns the possibly moved
/// list header after materialisation.
#[majit_macros::dont_look_inside]
pub unsafe fn w_list_materialize_range(obj: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    let obj = crate::gc_roots::pin_root(obj);
    let _list_guard = w_list_lock(obj);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let list = &mut *(obj as *mut W_ListObject);
    if matches!(
        list.strategy,
        ListStrategy::SimpleRange | ListStrategy::Range
    ) {
        switch_range_to_integer_strategy(list)
    } else {
        obj
    }
}

/// Replace a range strategy's immutable erased tuple.  Sharing means the old
/// tuple is never mutated; a pop publishes a fresh tuple on this list only.
unsafe fn install_range_state(
    obj: PyObjectRef,
    strategy: ListStrategy,
    values: &[i64],
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    let state_slot = crate::gc_roots::shadow_stack_len();
    let state = alloc_range_state(values);
    let _ = crate::gc_roots::pin_root(state as PyObjectRef);
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    list_write_barrier(obj);
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let list = &mut *(obj as *mut W_ListObject);
    list.strategy = strategy;
    list.items = crate::gc_roots::shadow_stack_get(state_slot) as *mut ItemsBlock;
    obj
}

/// listobject.py EmptyListStrategy.switch_to_correct_strategy.
///
/// First append on an empty list picks the typed strategy that matches
/// the appended item, then installs an empty typed storage. Caller is
/// expected to perform the actual append immediately afterward.
unsafe fn switch_to_correct_strategy(list: &mut W_ListObject, w_item: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::publish_roots(&[list as *mut W_ListObject as PyObjectRef, w_item]);
    crate::gc_roots::normalize_roots(root_base, 2);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let w_item = crate::gc_roots::shadow_stack_get(root_base + 1);
    let list = &mut *(obj as *mut W_ListObject);
    // SizeListStrategy.get_sizehint; EmptyListStrategy inherits the zero
    // default. The strategy object is replaced below, so consume the hint.
    let sizehint = if list.strategy == ListStrategy::Size {
        usize::try_from(sizehint_state_value(list.items)).unwrap_or(0)
    } else {
        0
    };
    list.set_length_relaxed(0);
    list.items = std::ptr::null_mut();
    if is_plain_int1(w_item) {
        let fresh = IntArray::with_capacity(sizehint);
        let obj = crate::gc_roots::shadow_stack_get(root_base);
        let list = &mut *(obj as *mut W_ListObject);
        list.int_items.install(fresh);
        let obj = crate::gc_roots::shadow_stack_get(root_base);
        let list = &mut *(obj as *mut W_ListObject);
        list.strategy = ListStrategy::Integer;
    } else if is_float_strategy_item(w_item) {
        let fresh = FloatArray::with_capacity(sizehint);
        let obj = crate::gc_roots::shadow_stack_get(root_base);
        let list = &mut *(obj as *mut W_ListObject);
        list.float_items.install(fresh);
        let obj = crate::gc_roots::shadow_stack_get(root_base);
        let list = &mut *(obj as *mut W_ListObject);
        list.strategy = ListStrategy::Float;
    } else if is_bytes_strategy_item(w_item) {
        // The immediately following append grows this null/zero rlist form
        // before storing. Avoiding a bulk Vec conversion here keeps the
        // generated append graph on PyPy's look-inside path.
        let fresh = BytesArray::with_capacity(sizehint);
        let obj = crate::gc_roots::shadow_stack_get(root_base);
        let _ = W_ListObject::install_bytes_items(obj, fresh);
        let obj = crate::gc_roots::shadow_stack_get(root_base);
        let list = &mut *(obj as *mut W_ListObject);
        list.strategy = ListStrategy::Bytes;
    } else if is_ascii_strategy_item(w_item) {
        let fresh = UnicodeArray::with_capacity(sizehint);
        let obj = crate::gc_roots::shadow_stack_get(root_base);
        let _ = W_ListObject::install_ascii_items(obj, fresh);
        let obj = crate::gc_roots::shadow_stack_get(root_base);
        let list = &mut *(obj as *mut W_ListObject);
        list.strategy = ListStrategy::Ascii;
    } else {
        let obj = crate::gc_roots::shadow_stack_get(root_base);
        let list = &mut *(obj as *mut W_ListObject);
        list.set_length_relaxed(0);
        list.items = std::ptr::null_mut();
        list.strategy = ListStrategy::Object;
        let _ = W_ListObject::object_resize_capacity(obj, sizehint);
    }
    crate::gc_roots::shadow_stack_get(root_base)
}

/// The strategy `w_list_new` picks for a given item set.
///
/// listobject.py EmptyListStrategy: a freshly created list with no
/// items uses Empty until first append picks a typed strategy.
pub fn list_strategy_for(items: &[PyObjectRef]) -> ListStrategy {
    if items.is_empty() {
        ListStrategy::Empty
    } else if all_ints(items) {
        ListStrategy::Integer
    } else if all_floats(items) {
        ListStrategy::Float
    } else if all_int_or_float(items) {
        ListStrategy::IntOrFloat
    } else if all_bytes(items) {
        ListStrategy::Bytes
    } else if all_ascii(items) {
        ListStrategy::Ascii
    } else {
        ListStrategy::Object
    }
}

/// Fire the GC write barrier for an Object-strategy list whose `items`
/// block just gained a possibly-young element. RPython's GC transform
/// emits `ll_writebarrier` (rgc.py) automatically after a pointer
/// store into a structure behind a custom tracer; pyre has no transform
/// pass, so the barrier runs here by hand. `list_object_custom_trace`
/// only forwards the off-GC `ItemsBlock` slots when the list is reached by
/// a collection; an old-gen list that stored a young element is reached on
/// a minor GC only if it sits in the remembered set, so the barrier must
/// run after every ref store. Mirrors `set_write_barrier` / `dict_write_barrier`.
///
/// `dont_look_inside`: the barrier is opaque to the JIT — the orthodox
/// append fold descends `w_list_append` and folds the store leaves to
/// native ops, but this barrier residualizes via the registered fnaddr so
/// the dropped-by-fold write barrier survives as a residual call (the off-GC
/// `ItemsBlock` is reached by the collector only through the remembered
/// `W_ListObject`, so the barrier must run for every appended ref).
#[majit_macros::dont_look_inside]
pub extern "C" fn list_write_barrier(obj: PyObjectRef) {
    list_write_barrier_impl(obj, false);
}

fn list_write_barrier_impl(obj: PyObjectRef, managed: bool) {
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let obj = crate::gc_roots::pin_root(obj);
    if managed {
        crate::gc_hook::try_gc_write_barrier_managed(obj as *mut u8);
    } else {
        crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
    }
    // Phase L2: when the block is a GC-managed array, the list-ptr forward in
    // `list_object_custom_trace` relocates a young block but does NOT re-scan an
    // already-old block's items — a young element just stored into an old block
    // would be missed. Barrier the block too so its varsize walker re-runs; the
    // collector no-ops the barrier on a still-young block (`TRACK_YOUNG_PTRS`
    // unset). Inert while the block stays std::alloc (`try_gc_owns_object` false).
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let list = unsafe { &*(obj as *const W_ListObject) };
    if list.strategy == ListStrategy::Object
        && !list.items.is_null()
        && crate::gc_hook::try_gc_owns_object(list.items as *mut u8)
    {
        // The ownership query is a safepoint.  Reload the field it may have
        // forwarded instead of handing the following rooted barrier the
        // pre-collection array address.
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        let items = unsafe { (*(obj as *const W_ListObject)).items };
        crate::gc_hook::try_gc_write_barrier(items as *mut u8);
    }
}

/// Generalize the items block's cards before its items are permuted in place,
/// and answer with the list at its post-barrier address.
///
/// A move stores no new reference, so `list_write_barrier` does not answer for
/// it: the referenced set is unchanged and only which card page holds each
/// pointer moves. A minor reaches a carded array through its dirty pages
/// alone, so a young pointer shifted into a clean page would not be scanned.
///
/// The ownership query is a safepoint and the barrier reads a header, so the
/// block cannot be handed over unchecked and the caller cannot keep its own
/// reference across the call — hence the root bracket here and the returned
/// address rather than a `()`.
#[majit_macros::dont_look_inside]
pub fn list_before_move_barrier(obj: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let _obj = crate::gc_roots::pin_root(obj);
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let list = unsafe { &*(obj as *const W_ListObject) };
    if list.strategy == ListStrategy::Object
        && !list.items.is_null()
        && crate::gc_hook::try_gc_owns_object(list.items as *mut u8)
    {
        // The ownership query is a safepoint. Reload the field it may have
        // forwarded instead of barriering the pre-collection array address.
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        let items = unsafe { (*(obj as *const W_ListObject)).items };
        crate::gc_hook::try_gc_write_barrier_before_move(items as *mut u8);
    }
    crate::gc_roots::shadow_stack_get(obj_slot)
}

/// Run the list's pre-write barrier while keeping the value in a relocatable
/// shadow-stack slot. A concurrent collector may run while the barrier waits
/// for the GC operation gate, so the raw Rust argument must be reloaded before
/// the following pointer store.
///
/// `dont_look_inside`: the `push_roots` bracket around the barrier resolves the
/// thread-local root stack and installs a `Drop` that truncates it — a
/// shadow-stack shape with no RPython counterpart (the GC transform emits
/// `ll_writebarrier` with no bracket at all) and no lowering in the tracer, so
/// the object-append descent used to hit its unregistered zero-arg callee as a
/// `symbolic_fnaddr_for_path` hash and decline the whole fold. Collapsing the
/// bracket and the barrier into one registered residual keeps the Object arm's
/// `set_len` / `setitem_fast` leaves foldable to native ops, and costs the same
/// one residual call the barrier alone already did.
///
/// Returns `value` at its post-barrier address: the ownership query inside
/// `list_write_barrier` is a safepoint, so the caller must store the returned
/// pointer rather than the argument it passed.
///
/// The signature spells `*mut PyObject` rather than the identical `PyObjectRef`
/// alias so that `emit_helper_call_target_fn` recognises the parameters and the
/// result as raw pointers and emits the `extern "C" fn(i64, i64) -> i64` call
/// trampoline. `jit_fnaddr.rs` registers that trampoline: a residual call's
/// target must carry the uniform word ABI, because the wasm backend lowers an
/// `Int`/`Ref`-result residual to a `call_indirect` whose static type comes from
/// the descr alone, and a raw `(*mut PyObject, *mut PyObject) -> *mut PyObject`
/// is `(i32, i32) -> i32` on wasm32.
#[majit_macros::dont_look_inside]
pub fn prepare_list_ref_store(obj: *mut PyObject, value: *mut PyObject) -> *mut PyObject {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::publish_roots(&[obj, value]);
    crate::gc_roots::normalize_roots(root_base, 2);
    list_write_barrier(crate::gc_roots::shadow_stack_get(root_base));
    crate::gc_roots::shadow_stack_get(root_base + 1)
}

/// Resolve a GC reference after a residual safepoint may have moved it.
/// RPython's GC transform reloads every livevar from the shadow stack; this is
/// the one-word residual form used by the descended list-append body.
#[majit_macros::dont_look_inside]
pub fn current_gc_ref(obj: *mut PyObject) -> *mut PyObject {
    let _roots = crate::gc_roots::push_roots();
    let obj = crate::gc_roots::pin_root(obj);
    crate::gc_hook::try_gc_current_object_address(obj as *mut u8) as *mut PyObject
}

/// Allocate a new W_ListObject from a Vec of items.
///
/// Residualized: the storage construction below
/// (`w_list_new_with_strategy`) drives the moving collector through
/// `push_roots` / `pin_root` / `alloc_list_items_block_gc` /
/// the collecting header allocator — shadow-stack and `box_assume_init` plumbing
/// the tracer cannot model. The JIT leaves the call as a residual
/// returning the fresh object pointer rather than tracing into the GC
/// allocator.
#[majit_macros::dont_look_inside]
pub fn w_list_new(items: Vec<PyObjectRef>) -> PyObjectRef {
    let strategy = list_strategy_for(&items);
    w_list_new_with_strategy(items, strategy)
}

/// Like `w_list_new` but pins the Object strategy regardless of contents, so
/// every element stays boxed by pointer identity (an all-int item set is NOT
/// unboxed into `int_items`). Used where element identity must survive, e.g. the
/// unpickler memo array.
///
/// Residualized for the same GC-allocator reason as `w_list_new`.
#[majit_macros::dont_look_inside]
pub fn w_list_new_object(items: Vec<PyObjectRef>) -> PyObjectRef {
    w_list_new_with_strategy(items, ListStrategy::Object)
}

/// Construct an empty list. Residualized so the `Vec::new()` backing-store
/// construction stays inside the opaque call rather than surfacing as a
/// separate residual funcptr at the caller's trace/blackhole level.
#[majit_macros::dont_look_inside]
pub fn w_list_new_empty() -> PyObjectRef {
    w_list_new_object(Vec::new())
}

/// `rpython.rlib.objectmodel.newlist_hint` for interpreter-level temporary
/// lists of wrapped objects.
///
/// This is deliberately an Object-strategy list, not a Size-strategy list:
/// `BaseObjSpace._unpackiterable_unknown_length` builds an RPython
/// `list[W_Root]`, whereas [`w_list_new_with_sizehint`] implements PyPy's
/// separate `space.newlist_hint` / `SizeListStrategy` object-space API.  An
/// unrepresentable allocation mirrors the upstream `except MemoryError:
/// items = []` fallback by returning an exact zero-capacity temporary.
#[majit_macros::dont_look_inside]
pub fn w_list_new_object_with_sizehint(sizehint: i64) -> PyObjectRef {
    let obj = w_list_new_object(Vec::new());
    let capacity = usize::try_from(sizehint)
        .ok()
        .filter(|&capacity| capacity <= (isize::MAX as usize) / std::mem::size_of::<PyObjectRef>())
        .unwrap_or(0);
    unsafe { W_ListObject::object_resize_capacity(obj, capacity) }
}

/// listobject.py `make_empty_list_with_size` / `SizeListStrategy`.
/// The hint belongs to the strategy while the list has no backing storage;
/// the first append consumes it when selecting the concrete strategy.
#[majit_macros::dont_look_inside]
pub fn w_list_new_with_sizehint(sizehint: i64) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let state_slot = crate::gc_roots::shadow_stack_len();
    let state = unsafe { alloc_sizehint_state(sizehint) };
    let _ = crate::gc_roots::pin_root(state as PyObjectRef);
    let obj = w_list_new_with_strategy(Vec::new(), ListStrategy::Size);
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    unsafe {
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        list_write_barrier(obj);
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        (*(obj as *mut W_ListObject)).items =
            crate::gc_roots::shadow_stack_get(state_slot) as *mut ItemsBlock;
        obj
    }
}

/// `listobject.py make_range_list`: build the erased immutable storage used by
/// `SimpleRangeListStrategy` or `RangeListStrategy` without materialising its
/// integer elements.
#[majit_macros::dont_look_inside]
pub fn w_list_new_range(start: i64, step: i64, length: i64) -> PyObjectRef {
    if length <= 0 {
        return w_list_new(Vec::new());
    }
    let (strategy, values): (ListStrategy, &[i64]) = if start == 0 && step == 1 {
        (ListStrategy::SimpleRange, std::slice::from_ref(&length))
    } else {
        (ListStrategy::Range, &[start, step, length])
    };
    let _roots = crate::gc_roots::push_roots();
    let state_slot = crate::gc_roots::shadow_stack_len();
    let state = unsafe { alloc_range_state(values) };
    let _ = crate::gc_roots::pin_root(state as PyObjectRef);
    let obj = w_list_new_with_strategy(Vec::new(), strategy);
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    unsafe {
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        list_write_barrier(obj);
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        (*(obj as *mut W_ListObject)).items =
            crate::gc_roots::shadow_stack_get(state_slot) as *mut ItemsBlock;
        obj
    }
}

/// Strategy clone for storage PyPy shares by identity: Size retains the same
/// mutable strategy instance, while both range strategies retain their
/// immutable erased tuple.  The check and clone are one list-locked operation
/// so a free-threaded mutation cannot switch the strategy between them.
#[majit_macros::dont_look_inside]
pub unsafe fn w_list_clone_if_shared_strategy(obj: PyObjectRef) -> Option<PyObjectRef> {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    let obj = crate::gc_roots::pin_root(obj);
    let _list_guard = w_list_lock(obj);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let list = &*(obj as *const W_ListObject);
    if !matches!(
        list.strategy,
        ListStrategy::Size | ListStrategy::SimpleRange | ListStrategy::Range
    ) {
        return None;
    }
    let strategy = list.strategy;
    let state_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(list.items as PyObjectRef);
    let clone = w_list_new_with_strategy(Vec::new(), strategy);
    let clone_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(clone);
    let clone = crate::gc_roots::shadow_stack_get(clone_slot);
    list_write_barrier(clone);
    let clone = crate::gc_roots::shadow_stack_get(clone_slot);
    (*(clone as *mut W_ListObject)).items =
        crate::gc_roots::shadow_stack_get(state_slot) as *mut ItemsBlock;
    Some(clone)
}

/// `SizeListStrategy` is the sole shared-storage strategy whose `mul` is a
/// no-op: it represents an empty list with a future allocation hint. Range
/// strategies instead inherit `ListStrategy.mul`, whose cloned receiver is
/// immediately materialised by `BaseRangeListStrategy.inplace_mul`.
#[majit_macros::dont_look_inside]
pub unsafe fn w_list_clone_if_size(obj: PyObjectRef) -> Option<PyObjectRef> {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    let obj = crate::gc_roots::pin_root(obj);
    let _list_guard = w_list_lock(obj);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let list = &*(obj as *const W_ListObject);
    if list.strategy != ListStrategy::Size {
        return None;
    }
    let state_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(list.items as PyObjectRef);
    let clone = w_list_new_with_strategy(Vec::new(), ListStrategy::Size);
    let clone_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(clone);
    let clone = crate::gc_roots::shadow_stack_get(clone_slot);
    list_write_barrier(clone);
    let clone = crate::gc_roots::shadow_stack_get(clone_slot);
    (*(clone as *mut W_ListObject)).items =
        crate::gc_roots::shadow_stack_get(state_slot) as *mut ItemsBlock;
    Some(clone)
}

/// Build the backing storage a `strategy`-strategy list holding `items` needs,
/// without installing it anywhere: the typed blocks (empty unless the matching
/// strategy) first, then the Object-strategy items block.
///
/// The caller must have pinned every element of `items` — the unboxing and the
/// block allocation below can both collect.  The returned block is young; the
/// caller must pin it across any further allocation before storing it.
///
/// Each typed block stays a bare Rust local until the caller installs it, so
/// every one of them is pinned across the allocations that follow it: old-gen
/// is mark-sweep, and an unrooted block with no heap edge yet is sweepable, not
/// merely immobile (`IntArray::pin_block`).  The caller owns the `push_roots`
/// scope those pins live in and closes the bracket with
/// [`ListStorage::reload_typed_blocks`] after its last allocation.
unsafe fn build_list_storage(items: &[PyObjectRef], strategy: ListStrategy) -> ListStorage {
    // Only the array matching `strategy` is ever read, so the other takes the
    // empty form rather than a block it will never address — the shape
    // `emit_typed_list_inline` / `emit_object_list_inline` already leave behind
    // for traced code, where the unused typed fields stay null from
    // `clear_gc_fields`.
    let int_items = match strategy {
        ListStrategy::Integer => {
            IntArray::from_vec(items.iter().map(|&item| plain_int_w(item)).collect())
        }
        ListStrategy::IntOrFloat => IntArray::from_vec(
            items
                .iter()
                .map(|&item| int_or_float_encode_item(item).unwrap())
                .collect(),
        ),
        _ => IntArray::empty(),
    };
    let int_block_root = int_items.pin_block();
    let float_items = if let ListStrategy::Float = strategy {
        FloatArray::from_vec(items.iter().map(|&item| w_float_get_value(item)).collect())
    } else {
        FloatArray::empty()
    };
    let float_block_root = float_items.pin_block();
    let bytes_items = if let ListStrategy::Bytes = strategy {
        BytesArray::from_vec(items.iter().map(|&item| w_bytes_block(item)).collect())
    } else {
        BytesArray::empty()
    };
    let bytes_block_root = bytes_items.pin_block();
    let ascii_items = if let ListStrategy::Ascii = strategy {
        UnicodeArray::from_vec(
            items
                .iter()
                .map(|&item| w_str_storage(item) as *const _)
                .collect(),
        )
    } else {
        UnicodeArray::empty()
    };
    let ascii_block_root = ascii_items.pin_block();
    let (length, block) = if let ListStrategy::Object = strategy {
        (items.len(), alloc_list_items_block_gc(items))
    } else {
        (0usize, std::ptr::null_mut())
    };
    ListStorage {
        length,
        block,
        int_items,
        float_items,
        bytes_items,
        ascii_items,
        int_block_root,
        float_block_root,
        bytes_block_root,
        ascii_block_root,
    }
}

/// Backing storage built but not yet installed, plus the shadow-stack slots the
/// typed blocks are pinned in.
struct ListStorage {
    length: usize,
    block: *mut ItemsBlock,
    int_items: IntArray,
    float_items: FloatArray,
    bytes_items: BytesArray,
    ascii_items: UnicodeArray,
    int_block_root: usize,
    float_block_root: usize,
    bytes_block_root: usize,
    ascii_block_root: usize,
}

impl ListStorage {
    /// The `pop_roots` half of [`IntArray::pin_block`]'s bracket: re-read both
    /// typed blocks once the caller's last allocation is behind them, since
    /// both take their heap edge from the store that follows.
    fn reload_typed_blocks(&mut self) {
        self.int_items.reload_block(self.int_block_root);
        self.float_items.reload_block(self.float_block_root);
        self.bytes_items.reload_block(self.bytes_block_root);
        self.ascii_items.reload_block(self.ascii_block_root);
    }
}

/// Construct a list with an explicitly selected storage strategy.
///
/// An interpreter-internal seam: Python-visible constructors go through
/// [`w_list_new`], which picks the strategy with [`list_strategy_for`]. The
/// JIT's strategy tests use this to pin one representation instead of
/// depending on that inference.
pub fn w_list_new_with_strategy(items: Vec<PyObjectRef>, strategy: ListStrategy) -> PyObjectRef {
    // `build_list_storage` unboxes each item the way `strategy` names, so a
    // strategy narrower than the items support reaches `plain_int_w` /
    // `w_float_get_value` on the wrong payload. A wider one is always sound —
    // `Object` keeps the pointers — so this admits every widening the seam
    // exists to express and only rejects the unboxing that has no payload.
    debug_assert!(
        match strategy {
            ListStrategy::Empty
            | ListStrategy::Size
            | ListStrategy::SimpleRange
            | ListStrategy::Range => items.is_empty(),
            ListStrategy::Integer => all_ints(&items),
            ListStrategy::Float => all_floats(&items),
            ListStrategy::IntOrFloat => all_int_or_float(&items),
            ListStrategy::Bytes => all_bytes(&items),
            ListStrategy::Ascii => all_ascii(&items),
            ListStrategy::Object => true,
        },
        "list items do not support the requested storage strategy",
    );
    // `gct_fv_gc_malloc` bracket pattern (`framework.py`):
    // pin every PyObjectRef in `items` before the GC malloc paths
    // below (`alloc_list_items_block_gc`, the collecting header allocation) so the
    // shadow stack walker sees them if a collection fires inside the
    // allocator. The Empty / Integer / Float / Bytes strategies still hold
    // PyObjectRef pointers in `items` until each element is unboxed
    // (`plain_int_w`, `w_float_get_value`); pinning all of them at
    // function entry covers every strategy uniformly.
    let _roots = crate::gc_roots::push_roots();
    for &item in &items {
        let _ = crate::gc_roots::pin_root(item);
    }

    // The nursery `items_block` is allocated last and pinned across the
    // collecting header allocation — the only later allocation that can
    // relocate it, since the typed-block allocs precede it.  The typed blocks
    // carry their own pins out of `build_list_storage`.
    let mut storage = unsafe { build_list_storage(&items, strategy) };
    let (length, mut items_block) = (storage.length, storage.block);
    // Phase L2: pin the (possibly young, GC-managed) items block across the
    // W_ListObject header allocation below may trigger a collection that
    // relocates the nursery block, so re-read its moved address
    // before storing it into the wrapper. Inert for a null or std::alloc block.
    let block_root: Option<usize> = if !items_block.is_null() {
        let s = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(items_block as PyObjectRef);
        Some(s)
    } else {
        None
    };
    let header = PyObject {
        ob_type: &LIST_TYPE as *const PyType,
        w_class: get_instantiate(&LIST_TYPE),
    };
    // rlist.py:116 `LIST = GcStruct(...)`: the list header is an ordinary
    // movable GC allocation, just like its `GcArray` items block.  Keeping the
    // header in old-gen would put every short-lived Object-strategy list in
    // `old_objects_pointing_to_young`; a minor collection must conservatively
    // trace those old headers and promote their item blocks, so a loop such as
    // `d = ["0"]` accumulates both allocations until the next major cycle.
    //
    // The items/typed block is the one GC child manufactured before its parent.
    // Pass it through the rooted collecting allocator, which performs
    // framework.py:853-856's collect-and-reserve bracket if the nursery is
    // full.  The surrounding shadow-stack roots remain authoritative for all
    // item values and reload every block below; `allocation_root` supplies the
    // allocator's direct translated-livevar slot as well.
    let mut allocation_root = match strategy {
        ListStrategy::Object => items_block as *mut u8,
        ListStrategy::Integer | ListStrategy::IntOrFloat => storage.int_items.block as *mut u8,
        ListStrategy::Float => storage.float_items.block as *mut u8,
        ListStrategy::Bytes => storage.bytes_items.block as *mut u8,
        ListStrategy::Ascii => storage.ascii_items.block as *mut u8,
        ListStrategy::Empty
        | ListStrategy::Size
        | ListStrategy::SimpleRange
        | ListStrategy::Range => std::ptr::null_mut(),
    };
    let mut needs_write_barrier = true;
    let raw = unsafe {
        crate::gc_hook::try_gc_alloc_collecting_rooted(
            W_LIST_GC_TYPE_ID,
            W_LIST_OBJECT_SIZE,
            &mut allocation_root,
            &mut needs_write_barrier,
        )
    };
    // The `std::alloc` header below is the answer for a missing hook, not for
    // a GC that owns the heap and refused: that one would leave the list's
    // managed edges untraced behind a header the collector never walks.
    let raw = crate::gc_hook::GcAllocOutcome::from_hook(raw)
        .allocated_or_abort(W_LIST_OBJECT_SIZE)
        .unwrap_or(std::ptr::null_mut());
    // `pop_roots` for the two typed blocks: this was the last allocation they
    // had to survive, and both take their heap edge from the struct built below.
    storage.reload_typed_blocks();
    let ListStorage {
        int_items,
        float_items,
        bytes_items,
        ascii_items,
        ..
    } = storage;
    // Re-read the (possibly relocated) nursery items block before either the
    // GC-owned or std::alloc fallback header takes ownership of the edge.
    if let Some(s) = block_root {
        items_block = crate::gc_roots::shadow_stack_get(s) as *mut ItemsBlock;
    }
    if raw.is_null() {
        let boxed = Box::new(W_ListObject {
            ob_header: header,
            allocated: items.len() as isize,
            length: AtomicUsize::new(length),
            items: items_block,
            strategy,
            int_items,
            float_items,
            bytes_items,
            ascii_items,
            w_slots: PY_NULL,
        });
        return Box::into_raw(boxed) as PyObjectRef;
    }
    unsafe {
        std::ptr::write(
            raw as *mut W_ListObject,
            W_ListObject {
                ob_header: header,
                allocated: items.len() as isize,
                length: AtomicUsize::new(length),
                items: items_block,
                strategy,
                int_items,
                float_items,
                bytes_items,
                ascii_items,
                w_slots: PY_NULL,
            },
        );
    }
    // A nursery header needs no creation barrier. The collecting allocator can
    // spill to old-gen (for example around pinned nursery gaps); only that
    // placement needs remembering for its young Object-strategy items edge.
    // Integer/Float blocks are old-gen leaf arrays and need no barrier.
    if matches!(
        strategy,
        ListStrategy::Object | ListStrategy::Bytes | ListStrategy::Ascii
    ) && needs_write_barrier
    {
        list_write_barrier_impl(raw as PyObjectRef, true);
    }
    raw as PyObjectRef
}

/// Read one app-level `__slots__` entry from a `list` subclass.
///
/// PyPy's `BaseUserClassMapdict.getslotvalue` indexes the instance-owned
/// storage list by `Member.index`. `PY_NULL` is the unbound-slot sentinel.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_slot_get(obj: PyObjectRef, index: usize) -> Option<PyObjectRef> {
    let slots = unsafe { (*(obj as *const W_ListObject)).w_slots };
    if slots.is_null() {
        return None;
    }
    unsafe { w_list_getitem(slots, index as i64) }.filter(|value| !value.is_null())
}

/// Write one app-level `__slots__` entry on a `list` subclass.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_slot_set(obj: PyObjectRef, index: usize, value: PyObjectRef) {
    let _roots = crate::gc_roots::push_roots();
    let _ = crate::gc_roots::pin_root(obj);
    let _ = crate::gc_roots::pin_root(value);
    let obj_slot = crate::gc_roots::shadow_stack_len() - 2;
    let value_slot = crate::gc_roots::shadow_stack_len() - 1;

    let mut rooted_obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let mut slots = unsafe { (*(rooted_obj as *const W_ListObject)).w_slots };
    if slots.is_null() {
        slots = w_list_new(vec![PY_NULL; index + 1]);
        rooted_obj = crate::gc_roots::shadow_stack_get(obj_slot);
        unsafe { (*(rooted_obj as *mut W_ListObject)).w_slots = slots };
        crate::gc_hook::try_gc_write_barrier(rooted_obj as *mut u8);
    }
    let _ = crate::gc_roots::pin_root(slots);
    let slots_slot = crate::gc_roots::shadow_stack_len() - 1;
    while unsafe { w_list_len(crate::gc_roots::shadow_stack_get(slots_slot)) } <= index {
        unsafe { w_list_append(crate::gc_roots::shadow_stack_get(slots_slot), PY_NULL) };
    }
    unsafe {
        w_list_setitem(
            crate::gc_roots::shadow_stack_get(slots_slot),
            index as i64,
            crate::gc_roots::shadow_stack_get(value_slot),
        );
    }
}

/// Clear one app-level `__slots__` entry on a `list` subclass.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_slot_del(obj: PyObjectRef, index: usize) -> bool {
    let slots = unsafe { (*(obj as *const W_ListObject)).w_slots };
    if slots.is_null()
        || unsafe { w_list_getitem(slots, index as i64) }.is_none_or(|value| value.is_null())
    {
        return false;
    }
    unsafe { w_list_setitem(slots, index as i64, PY_NULL) }
}

// Integer-strategy low-level access primitives, mirroring the rlist.py
// oopspec leaves the JIT codewriter substitutes for clean array operations.
// The runtime bodies below are the fallback when the call is not looked into;
// the codewriter recognises the `#[oopspec("list.int_*")]` tag and emits
// `GetfieldGcR(int_items.block) → GetarrayitemGcI` / `SetarrayitemGcI` /
// `GetfieldGcI(int_items.len)` instead (see int_array.rs). The `_fast`
// accessors take a non-negative, in-bounds `index`; the caller normalises
// negative indices and checks the bound, as `ll_getitem`/`ll_setitem` wrap
// `ll_getitem_fast`/`ll_setitem_fast` in rlist.py.

/// `ll_length` for the Integer strategy (rlist.py `'list.len(l)'`).
#[majit_macros::oopspec("list.int_len(l)")]
pub fn ll_list_int_length(l: &W_ListObject) -> usize {
    l.int_items.len()
}

/// `ll_getitem_fast` for the Integer strategy (rlist.py:375
/// `'list.getitem(l, index)'`): raw unboxed read at a known-in-bounds index.
#[majit_macros::oopspec("list.int_getitem(l, index)")]
pub fn ll_list_int_getitem_fast(l: &W_ListObject, index: usize) -> i64 {
    l.int_items.as_slice()[index]
}

/// `ll_setitem_fast` for the Integer strategy (rlist.py
/// `'list.setitem(l, index, item)'`): raw unboxed write at a known-in-bounds
/// index.
#[majit_macros::oopspec("list.int_setitem(l, index, item)")]
pub fn ll_list_int_setitem_fast(l: &mut W_ListObject, index: usize, item: i64) {
    l.int_items.as_mut_slice()[index] = item;
}

/// Allocated capacity for the Integer strategy. `ll_append`'s resize-ge
/// fast case (rlist.py:285) inlines the append only while
/// `len(items) >= length + 1`, i.e. spare capacity exists.
#[majit_macros::oopspec("list.int_capacity(l)")]
pub fn ll_list_int_capacity(l: &W_ListObject) -> usize {
    l.int_items.heap_capacity()
}

/// Store the Integer-strategy live length (`_ll_list_resize_ge`'s
/// `l.length = newsize`, rlist.py:293). The caller has already ensured
/// the block has room, so this only bumps the length field.
#[majit_macros::oopspec("list.int_set_len(l, n)")]
pub fn ll_list_int_set_len(l: &mut W_ListObject, n: usize) {
    l.int_items.set_len(n);
}

// Float-strategy storage leaves, mirroring the Integer leaves above but
// addressing `float_items.{len,block}` and holding unboxed `f64` scalars.
// The codewriter recognises the `#[oopspec("list.float_*")]` tag and emits
// `GetfieldGcR(float_items.block) → GetarrayitemGcF` / `SetarrayitemGcF` /
// `GetfieldGcI(float_items.len)` (see float_array.rs).

/// `ll_length` for the Float strategy (rlist.py `'list.len(l)'`).
#[majit_macros::oopspec("list.float_len(l)")]
pub fn ll_list_float_length(l: &W_ListObject) -> usize {
    l.float_items.len()
}

/// `ll_setitem_fast` for the Float strategy (rlist.py
/// `'list.setitem(l, index, item)'`): raw unboxed write at a known-in-bounds
/// index.
#[majit_macros::oopspec("list.float_setitem(l, index, item)")]
pub fn ll_list_float_setitem_fast(l: &mut W_ListObject, index: usize, item: f64) {
    l.float_items.as_mut_slice()[index] = item;
}

/// Allocated capacity for the Float strategy. `ll_append`'s resize-ge
/// fast case (rlist.py:285) inlines the append only while
/// `len(items) >= length + 1`, i.e. spare capacity exists.
#[majit_macros::oopspec("list.float_capacity(l)")]
pub fn ll_list_float_capacity(l: &W_ListObject) -> usize {
    l.float_items.heap_capacity()
}

/// Store the Float-strategy live length (`_ll_list_resize_ge`'s
/// `l.length = newsize`, rlist.py:293). The caller has already ensured
/// the block has room, so this only bumps the length field.
#[majit_macros::oopspec("list.float_set_len(l, n)")]
pub fn ll_list_float_set_len(l: &mut W_ListObject, n: usize) {
    l.float_items.set_len(n);
}

// Object-strategy storage leaves, mirroring the Integer leaves above but
// addressing the `length` header + the `items` GcArray block (`Ptr(GcArray
// (OBJECTPTR))`). The element is a GC pointer, so the store carries the
// list write barrier — the only structural difference from the unboxed
// Integer/Float scalar stores.

/// `ll_length` for the Object strategy: the live `length` header
/// (rlist.py:116 `l.length`).
#[majit_macros::oopspec("list.obj_len(l)")]
pub fn ll_list_obj_length(l: &W_ListObject) -> usize {
    l.length_relaxed()
}

/// Allocated capacity for the Object strategy — the `items` block's
/// offset-0 GcArray length header (rlist.py:251 `len(l.items)`).
#[majit_macros::oopspec("list.obj_capacity(l)")]
pub fn ll_list_obj_capacity(l: &W_ListObject) -> usize {
    unsafe { items_block_capacity(l.items) }
}

/// Store the Object-strategy live length (`_ll_list_resize_ge`'s
/// `l.length = newsize`, rlist.py:293).
#[majit_macros::oopspec("list.obj_set_len(l, n)")]
pub fn ll_list_obj_set_len(l: &mut W_ListObject, n: usize) {
    l.set_length_relaxed(n);
}

/// `ll_getitem_fast` for the Object strategy: a GC-ref read at a
/// known-in-bounds index (`ll_pop_default`'s read, rlist.py).
#[majit_macros::oopspec("list.obj_getitem(l, index)")]
pub fn ll_list_obj_getitem_fast(l: &W_ListObject, index: usize) -> PyObjectRef {
    unsafe {
        let base = items_block_items_base(l.items);
        *base.add(index)
    }
}

/// `ll_setitem_fast` for the Object strategy: a GC-ref store at a
/// known-in-bounds index (the spare-capacity append's element write).
/// The element is a GC pointer, but — unlike the runtime helper that once
/// inlined the barrier here — the list write barrier is run by the caller
/// (`w_list_append`) as a separate `dont_look_inside` call. The orthodox
/// fold replaces this leaf with `getfield_gc_r(items) + setarrayitem_gc_r`
/// and would drop an inlined barrier; keeping the barrier in the caller
/// lets the fold preserve it as a residual call.
#[majit_macros::oopspec("list.obj_setitem(l, index, item)")]
pub fn ll_list_obj_setitem_fast(l: &mut W_ListObject, index: usize, item: PyObjectRef) {
    unsafe {
        let base = items_block_items_base(l.items);
        *base.add(index) = item;
    }
}

/// Get the item at the given index from a list.
///
/// Supports negative indexing. Returns None if out of bounds.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_getitem(obj: PyObjectRef, index: i64) -> Option<PyObjectRef> {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    let obj = crate::gc_roots::pin_root(obj);
    let _list_guard = w_list_lock(obj);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let list = &*(obj as *const W_ListObject);
    match list.strategy {
        // listobject.py EmptyListStrategy.getitem raises IndexError.
        ListStrategy::Empty | ListStrategy::Size => None,
        ListStrategy::SimpleRange | ListStrategy::Range => {
            let len = range_list_length(list) as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return None;
            }
            Some(w_int_new(range_list_item_unchecked(list, idx as usize)))
        }
        ListStrategy::Object => {
            let len = list.length_relaxed() as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return None;
            }
            let base = items_block_items_base(list.items);
            Some(*base.add(idx as usize))
        }
        ListStrategy::Integer => {
            let len = ll_list_int_length(list) as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return None;
            }
            Some(w_int_new(ll_list_int_getitem_fast(list, idx as usize)))
        }
        ListStrategy::IntOrFloat => {
            let items = list.int_items.as_slice();
            let len = items.len() as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return None;
            }
            let value = items[idx as usize];
            Some(if int_or_float_is_int(value) {
                w_int_new(int_or_float_decode_int(value))
            } else {
                w_float_new(f64::from_bits(value as u64))
            })
        }
        ListStrategy::Float => {
            let items = list.float_items.as_slice();
            let len = items.len() as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return None;
            }
            Some(w_float_new(items[idx as usize]))
        }
        ListStrategy::Bytes => {
            let len = list.bytes_items.len() as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return None;
            }
            Some(w_bytes_from_block(list.bytes_items[idx as usize]))
        }
        ListStrategy::Ascii => {
            let len = list.ascii_items.len() as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return None;
            }
            Some(w_str_from_storage(list.ascii_items[idx as usize] as *mut _))
        }
    }
}

/// Set the item at the given index in a list.
///
/// Supports negative indexing. Returns false if out of bounds.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_setitem(obj: PyObjectRef, index: i64, value: PyObjectRef) -> bool {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::publish_roots(&[obj, value]);
    crate::gc_roots::normalize_roots(root_base, 2);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let _list_guard = w_list_lock(obj);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let value = crate::gc_roots::shadow_stack_get(root_base + 1);
    let list = &mut *(obj as *mut W_ListObject);
    match list.strategy {
        // listobject.py EmptyListStrategy.setitem raises IndexError.
        ListStrategy::Empty | ListStrategy::Size => false,
        ListStrategy::SimpleRange | ListStrategy::Range => {
            let obj = switch_range_to_integer_strategy(list);
            w_list_setitem(obj, index, crate::gc_roots::shadow_stack_get(root_base + 1))
        }
        ListStrategy::Object => {
            let len = list.length_relaxed() as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return false;
            }
            let value = prepare_list_ref_store(obj, value);
            let obj = crate::gc_roots::shadow_stack_get(root_base);
            let list = &mut *(obj as *mut W_ListObject);
            let base = items_block_items_base(list.items);
            *base.add(idx as usize) = value;
            true
        }
        ListStrategy::Integer => {
            let len = ll_list_int_length(list) as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return false;
            }
            // AbstractUnwrappedStrategy.setitem (listobject.py): plain_int_w (unwrap)
            if is_plain_int1(value) {
                ll_list_int_setitem_fast(list, idx as usize, plain_int_w(value));
                true
            } else if is_float_strategy_item(value) && integer_to_int_or_float(list) {
                w_list_setitem(
                    crate::gc_roots::shadow_stack_get(root_base),
                    index,
                    crate::gc_roots::shadow_stack_get(root_base + 1),
                )
            } else {
                switch_to_object_strategy(list);
                w_list_setitem(
                    crate::gc_roots::shadow_stack_get(root_base),
                    index,
                    crate::gc_roots::shadow_stack_get(root_base + 1),
                )
            }
        }
        ListStrategy::IntOrFloat => {
            let len = list.int_items.len() as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return false;
            }
            if let Some(value) = int_or_float_encode_item(value) {
                list.int_items[idx as usize] = value;
                true
            } else {
                switch_to_object_strategy(list);
                w_list_setitem(
                    crate::gc_roots::shadow_stack_get(root_base),
                    index,
                    crate::gc_roots::shadow_stack_get(root_base + 1),
                )
            }
        }
        ListStrategy::Float => {
            let len = list.float_items.len() as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return false;
            }
            if is_float_strategy_item(value) {
                list.float_items[idx as usize] = w_float_get_value(value);
                true
            } else if is_plain_int1(value)
                && int_or_float_encode_int(plain_int_w(value)).is_some()
                && float_to_int_or_float(list)
            {
                w_list_setitem(
                    crate::gc_roots::shadow_stack_get(root_base),
                    index,
                    crate::gc_roots::shadow_stack_get(root_base + 1),
                )
            } else {
                switch_to_object_strategy(list);
                w_list_setitem(
                    crate::gc_roots::shadow_stack_get(root_base),
                    index,
                    crate::gc_roots::shadow_stack_get(root_base + 1),
                )
            }
        }
        ListStrategy::Bytes => {
            let len = list.bytes_items.len() as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return false;
            }
            if is_bytes_strategy_item(value) {
                list.bytes_items.set(idx as usize, w_bytes_block(value));
                true
            } else {
                switch_to_object_strategy(list);
                w_list_setitem(
                    crate::gc_roots::shadow_stack_get(root_base),
                    index,
                    crate::gc_roots::shadow_stack_get(root_base + 1),
                )
            }
        }
        ListStrategy::Ascii => {
            let len = list.ascii_items.len() as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return false;
            }
            if is_ascii_strategy_item(value) {
                list.ascii_items
                    .set(idx as usize, w_str_storage(value) as *const _);
                true
            } else {
                switch_to_object_strategy(list);
                w_list_setitem(
                    crate::gc_roots::shadow_stack_get(root_base),
                    index,
                    crate::gc_roots::shadow_stack_get(root_base + 1),
                )
            }
        }
    }
}

/// Append an item to a list.
///
/// Splits into a guard-taking wrapper and a lock-free
/// [`w_list_append_inner`], the same shape the dict side uses
/// (`w_dict_store_checked` / `w_dict_store_checked_inner`), because the append
/// fold descends this body:
///
/// * the wrapper must stay look-inside — the codewriter only reaches graphs
///   through look-inside calls from a jitdriver portal
///   (`grab_initial_jitcodes` / `enum_pending_graphs`), so a
///   `dont_look_inside` wrapper hides the inner body from the pipeline as
///   well;
/// * the descended body must hold no guard — a `w_list_lock` acquire/release
///   pair inside it declines the fold's sub-walk.
///
/// Either way `list_append_jitcode()` resolves to `None`, the fold declines,
/// and every `list.append` becomes a `Void` residual — a body effect that
/// refuses in-flight FOR_ITER delivery and silently drops the iteration.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_append(obj: PyObjectRef, value: PyObjectRef) {
    // PyPy executes this body under the GIL.  Pyre's free-threaded list lock
    // may enter `before_external_block`, which is a GC safepoint; preserve
    // the two RPython livevars across that extra boundary and reload them
    // before entering the lock-free strategy body.
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::publish_roots(&[obj, value]);
    crate::gc_roots::normalize_roots(root_base, 2);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let _list_guard = w_list_lock(obj);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let value = crate::gc_roots::shadow_stack_get(root_base + 1);
    let list = &mut *(obj as *mut W_ListObject);
    let old_size = list.live_len();
    w_list_append_inner(obj, value);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let list = &mut *(obj as *mut W_ListObject);
    list.sync_allocated(old_size);
}

/// [`w_list_append`], answering the index the value landed at, read under the
/// same guard as the append itself.
///
/// A caller that needs that index cannot append and then call [`w_list_len`]:
/// the append releases the lock before the length reacquires it, so a second
/// appender can land in between, and both callers then read the same length
/// and claim the same slot.
///
/// The body is spelled out rather than shared with [`w_list_append`], whose
/// exact shape the append fold descends -- routing that wrapper through a
/// further call resolves `list_append_jitcode()` to `None` and turns every
/// `list.append` into a `Void` residual.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_append_returning_index(obj: PyObjectRef, value: PyObjectRef) -> usize {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::publish_roots(&[obj, value]);
    crate::gc_roots::normalize_roots(root_base, 2);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let _list_guard = w_list_lock(obj);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let value = crate::gc_roots::shadow_stack_get(root_base + 1);
    let list = &mut *(obj as *mut W_ListObject);
    let old_size = list.live_len();
    w_list_append_inner(obj, value);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let list = &mut *(obj as *mut W_ListObject);
    list.sync_allocated(old_size);
    old_size
}

/// CPython `list_extend_iter_lock_held`'s direct-store append while its
/// length-hint reservation still has a free logical slot. The physical
/// strategy append is identical; only `list_resize` is skipped.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_append_preallocated(obj: PyObjectRef, value: PyObjectRef) {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::publish_roots(&[obj, value]);
    crate::gc_roots::normalize_roots(root_base, 2);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let _list_guard = w_list_lock(obj);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let value = crate::gc_roots::shadow_stack_get(root_base + 1);
    let list = &mut *(obj as *mut W_ListObject);
    let old_size = list.live_len();
    let old_allocated = list.allocated;
    w_list_append_inner(obj, value);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let list = &mut *(obj as *mut W_ListObject);
    if old_allocated > old_size as isize {
        list.allocated = old_allocated;
    } else {
        list.sync_allocated(old_size);
    }
}

/// [`w_list_append`]'s body, run with the list's guard already held.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`, and the caller must hold
/// `w_list_lock(obj)`.
pub unsafe fn w_list_append_inner(obj: PyObjectRef, value: PyObjectRef) {
    let list = &mut *(obj as *mut W_ListObject);
    match list.strategy {
        // listobject.py EmptyListStrategy.append: pick the matching
        // typed strategy first, then fall through to its append.
        ListStrategy::Empty | ListStrategy::Size => {
            let obj = switch_to_correct_strategy(list, value);
            let value = current_gc_ref(value);
            w_list_append_inner(obj, value);
        }
        ListStrategy::SimpleRange | ListStrategy::Range => {
            let obj = if is_plain_int1(value) {
                switch_range_to_integer_strategy(list)
            } else {
                switch_to_object_strategy(list)
            };
            let value = current_gc_ref(value);
            w_list_append_inner(obj, value);
        }
        // AbstractUnwrappedStrategy.append (listobject.py):
        //   if self.is_correct_type(w_item): l.append(self.unwrap(w_item)); return
        //   self.switch_to_next_strategy(w_list, w_item); w_list.append(w_item)
        ListStrategy::Object => {
            // ll_append (rlist.py) resize-ge fast case (rlist.py):
            // store in place while there is spare capacity (bump the length
            // and write the GC ref); otherwise fall back to the resizing
            // push. The element is a GC pointer, so the in-place store runs
            // the list write barrier after the store — a separate
            // `dont_look_inside` call the orthodox fold keeps residual while
            // the `set_len` / `setitem` leaves fold to native ops.
            let length = ll_list_obj_length(list);
            if length < ll_list_obj_capacity(list) {
                let value = prepare_list_ref_store(obj, value);
                let obj = current_gc_ref(obj);
                let list = &mut *(obj as *mut W_ListObject);
                ll_list_obj_set_len(list, length + 1);
                ll_list_obj_setitem_fast(list, length, value);
            } else {
                list.object_push(value);
            }
        }
        ListStrategy::Integer => {
            if is_plain_int1(value) {
                // ll_append (rtyper/rlist.py): length = ll_length();
                // _ll_resize_ge(length+1); ll_setitem_fast(length, item).
                // The resize-ge fast case (rlist.py:285) inlines only while
                // there is spare capacity; bump the length and store in
                // place. Otherwise fall back to the resizing push.
                let item = plain_int_w(value);
                let length = ll_list_int_length(list);
                if length < ll_list_int_capacity(list) {
                    ll_list_int_set_len(list, length + 1);
                    ll_list_int_setitem_fast(list, length, item);
                } else {
                    list.int_items.push(item);
                }
            } else if is_float_strategy_item(value) && integer_to_int_or_float(list) {
                let obj = current_gc_ref(obj);
                let value = current_gc_ref(value);
                w_list_append_inner(obj, value);
            } else {
                let obj = switch_to_object_strategy(list);
                let value = current_gc_ref(value);
                let list = &mut *(obj as *mut W_ListObject);
                list.object_push(value);
            }
        }
        ListStrategy::Float => {
            // `FloatListStrategy.is_correct_type` (listobject.py) is
            // `type(w_obj) is W_FloatObject` — a strict identity check that
            // rejects float subclasses (which share `ob_type == &FLOAT_TYPE`
            // but overwrite `w_class`), matching the Integer arm's
            // `is_plain_int1`.  A subclass de-specialises to Object storage
            // rather than being stored unboxed (which would lose its identity).
            if is_float_strategy_item(value) {
                // ll_append (rtyper/rlist.py): length = ll_length();
                // _ll_resize_ge(length+1); ll_setitem_fast(length, item). The
                // resize-ge fast case (rlist.py:285) inlines only while there
                // is spare capacity; bump the length and store in place.
                // Otherwise fall back to the resizing push.
                let item = w_float_get_value(value);
                let length = ll_list_float_length(list);
                if length < ll_list_float_capacity(list) {
                    ll_list_float_set_len(list, length + 1);
                    ll_list_float_setitem_fast(list, length, item);
                } else {
                    list.float_items.push(item);
                }
            } else if is_plain_int1(value)
                && int_or_float_encode_int(plain_int_w(value)).is_some()
                && float_to_int_or_float(list)
            {
                let obj = current_gc_ref(obj);
                let value = current_gc_ref(value);
                w_list_append_inner(obj, value);
            } else {
                let obj = switch_to_object_strategy(list);
                let value = current_gc_ref(value);
                let list = &mut *(obj as *mut W_ListObject);
                list.object_push(value);
            }
        }
        ListStrategy::IntOrFloat => {
            if let Some(item) = int_or_float_encode_item(value) {
                list.int_items.push(item);
            } else {
                let obj = switch_to_object_strategy(list);
                let value = current_gc_ref(value);
                let list = &mut *(obj as *mut W_ListObject);
                list.object_push(value);
            }
        }
        ListStrategy::Bytes => {
            if is_bytes_strategy_item(value) {
                let value = prepare_list_ref_store(obj, value);
                let obj = current_gc_ref(obj);
                let list = &*(obj as *const W_ListObject);
                // At capacity, route the grow through the list the way
                // `object_push` does: the fresh block reaches `bytes_items`
                // with the owner barrier directly in front of the store.
                let value = if list.bytes_items.spare_capacity() == 0 {
                    w_list_grow_bytes_block(obj, value)
                } else {
                    value
                };
                let obj = current_gc_ref(obj);
                let list = &mut *(obj as *mut W_ListObject);
                list.bytes_items.push(w_bytes_block(value));
            } else {
                let obj = switch_to_object_strategy(list);
                let value = current_gc_ref(value);
                let list = &mut *(obj as *mut W_ListObject);
                list.object_push(value);
            }
        }
        ListStrategy::Ascii => {
            if is_ascii_strategy_item(value) {
                let value = prepare_list_ref_store(obj, value);
                let obj = current_gc_ref(obj);
                let list = &*(obj as *const W_ListObject);
                let value = if list.ascii_items.spare_capacity() == 0 {
                    w_list_grow_ascii_block(obj, value)
                } else {
                    value
                };
                let obj = current_gc_ref(obj);
                let list = &mut *(obj as *mut W_ListObject);
                list.ascii_items.push(w_str_storage(value) as *const _);
            } else {
                let obj = switch_to_object_strategy(list);
                let value = current_gc_ref(value);
                let list = &mut *(obj as *mut W_ListObject);
                list.object_push(value);
            }
        }
    }
}

/// Drain-only `dont_look_inside` seam over [`w_list_append`].
///
/// The jd1 `_unpackiterable_unknown_length` driver compiles its drain loop and
/// blackhole-executes it on a guard failure. An *inlined* append would surface
/// each of its strategy/grow helpers (`object_push`,
/// `switch_to_correct_strategy`, typed-array grow, …) as a separate residual
/// funcptr the blackhole must resolve; wrapping the drain's append in one
/// `dont_look_inside` residual collapses that whole subtree to a single
/// registered address (`jit_fnaddr.rs`) — the same seam the drain already draws
/// around its `w_list_new_empty` prologue and `drain_collect_items` epilogue.
/// The global `list.append` path keeps calling [`w_list_append`] directly and
/// stays traced, so the append fold and the escape-flush replay are unaffected.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
///
/// `#[inline(never)]` is load-bearing and separate from the tracing policy: the
/// body forwards verbatim to [`w_list_append`], so inlining the callee makes the
/// two functions byte-identical, and both are registered residual-call targets
/// (`jit_fnaddr.rs`). A linker that folds identical code — MSVC's `/OPT:ICF`,
/// on by default — then gives them one address, and `runtime_fnaddr_patch`
/// cannot tell which callee a `constants_i` entry meant. Keeping the forwarding
/// call keeps the two bodies distinct.
#[inline(never)]
#[majit_macros::dont_look_inside]
pub unsafe fn drain_list_append(obj: PyObjectRef, value: PyObjectRef) {
    w_list_append(obj, value)
}

/// Uniform residual-call ABI adapter for [`drain_list_append`].
///
/// Wasm function-table entries retain their physical parameter types, so the
/// raw pointer signature above is `(i32, i32) -> ()` on wasm32. JIT residual
/// calls carry Int and Ref operands in i64 locals; registering this adapter
/// gives the table target the same `(i64, i64) -> ()` signature emitted by the
/// wasm backend.
#[inline(never)]
pub extern "C" fn jit_drain_list_append(obj: i64, value: i64) {
    unsafe { drain_list_append(obj as PyObjectRef, value as PyObjectRef) }
}

/// Set the live length of an Integer-strategy list without reallocating
/// or boxing — the undo of a spare-capacity append (`_ll_list_resize_ge`'s
/// `l.length = newsize` run in reverse).  The backing array already has
/// room (the append that this reverses was admitted by
/// [`w_list_can_append_without_realloc`]), so this only rewinds the length
/// field.
///
/// # Safety
/// `obj` must point to a valid Integer-strategy `W_ListObject` whose
/// backing array has capacity for at least `n` elements.
pub unsafe fn w_list_int_set_len(obj: PyObjectRef, n: usize) {
    let list = &mut *(obj as *mut W_ListObject);
    debug_assert_eq!(
        list.strategy,
        ListStrategy::Integer,
        "w_list_int_set_len on non-Integer strategy"
    );
    ll_list_int_set_len(list, n);
}

/// JIT rollback leaves for IntOrFloatListStrategy's signed-longlong storage.
/// These mirror the Integer leaves but encode the restored boxed value using
/// listobject.py `IntOrFloatListStrategy.unwrap`.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_int_or_float_set_len(obj: PyObjectRef, n: usize) {
    let list = &mut *(obj as *mut W_ListObject);
    debug_assert_eq!(list.strategy, ListStrategy::IntOrFloat);
    list.int_items.set_len(n);
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_int_or_float_setitem(
    obj: PyObjectRef,
    index: usize,
    value: PyObjectRef,
) -> bool {
    let list = &mut *(obj as *mut W_ListObject);
    debug_assert_eq!(list.strategy, ListStrategy::IntOrFloat);
    let Some(value) = int_or_float_encode_item(value) else {
        return false;
    };
    list.int_items[index] = value;
    true
}

/// Get the length of a list: `W_ListObject.length()`, a strategy-dispatched
/// field read (`listobject.py EmptyListStrategy.length` returns 0).
///
/// Read without the list's lock, the way `PyList_GET_SIZE` reads `ob_size`
/// and the way every compiled read of it already did: the `builtin_len`
/// fold records `guard_value(strategy)` + `getfield(length)` and the walked
/// `w_list_*_inner` bodies hold no guard.  The value only ever becomes an
/// int, so a read that races a strategy switch yields a stale length, never
/// an out-of-bounds access.  With no lock there is no collection point, so
/// `obj` needs no root here.  Taking the lock instead made the acquire an
/// opaque residual (`w_list_lock` is `dont_look_inside`) on every traced
/// `len(list)`, which is what kept the generic builtin descent out of `len`.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_len(obj: PyObjectRef) -> usize {
    (*(obj as *const W_ListObject)).live_len()
}

/// CPython-visible `PyListObject.allocated` under the list's mutation lock.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_allocated(obj: PyObjectRef) -> isize {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    let obj = crate::gc_roots::pin_root(obj);
    let _list_guard = w_list_lock(obj);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    (*(obj as *const W_ListObject)).allocated
}

/// listobject.py `ListStrategy.physical_size` and the strategy-specific
/// overrides used by `__pypy__.list_get_physical_size`.
pub unsafe fn w_list_physical_size(obj: PyObjectRef) -> Option<usize> {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    let obj = crate::gc_roots::pin_root(obj);
    let _list_guard = w_list_lock(obj);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let list = &*(obj as *const W_ListObject);
    match list.strategy {
        ListStrategy::Empty | ListStrategy::Size => Some(0),
        ListStrategy::Object => Some(list.object_items_capacity()),
        ListStrategy::Integer | ListStrategy::IntOrFloat => Some(list.int_items.heap_capacity()),
        ListStrategy::Float => Some(list.float_items.heap_capacity()),
        ListStrategy::Bytes => Some(list.bytes_items.heap_capacity()),
        ListStrategy::Ascii => Some(list.ascii_items.heap_capacity()),
        // BaseRangeListStrategy inherits ListStrategy.physical_size, whose
        // diagnostic contract is to raise rather than invent an allocation.
        ListStrategy::SimpleRange | ListStrategy::Range => None,
    }
}

/// `W_ListObject._resize_hint` → strategy `_resize_hint`.
///
/// Returns false only when the RPython over-allocation arithmetic cannot be
/// represented. A shrink hint is clamped to the live length: translated PyPy
/// permits a lying private hint to truncate the backing below live elements,
/// but Rust slices require the same `length <= capacity` invariant that the
/// real caller is documented to uphold.
pub unsafe fn w_list_resize_hint(obj: PyObjectRef, newsize: i64) -> bool {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    let obj = crate::gc_roots::pin_root(obj);
    let _list_guard = w_list_lock(obj);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let list = &mut *(obj as *mut W_ListObject);
    match list.strategy {
        ListStrategy::Empty => {
            if newsize != 0 {
                let Ok(newsize) = isize::try_from(newsize) else {
                    return false;
                };
                let state_slot = crate::gc_roots::shadow_stack_len();
                let state = alloc_sizehint_state(newsize as i64);
                let _ = crate::gc_roots::pin_root(state as PyObjectRef);
                let obj = crate::gc_roots::shadow_stack_get(root_base);
                list_write_barrier(obj);
                let obj = crate::gc_roots::shadow_stack_get(root_base);
                let list = &mut *(obj as *mut W_ListObject);
                list.strategy = ListStrategy::Size;
                list.items = crate::gc_roots::shadow_stack_get(state_slot) as *mut ItemsBlock;
            }
            return true;
        }
        ListStrategy::Size => {
            let Ok(newsize) = isize::try_from(newsize) else {
                return false;
            };
            set_sizehint_state_value(list.items, newsize as i64);
            return true;
        }
        // BaseRangeListStrategy._resize_hint: supported as a deliberate no-op.
        ListStrategy::SimpleRange | ListStrategy::Range => return true,
        _ => {}
    }

    let newsize = usize::try_from(newsize).unwrap_or(0);

    let current = strategy_capacity(list).expect("strategy carries a backing array");
    let requested = newsize.max(list.live_len());
    let target = if requested > current {
        let extra = if requested < 9 { 3 } else { 6 };
        let Some(target) = requested
            .checked_add(extra)
            .and_then(|n| n.checked_add(requested >> 3))
        else {
            return false;
        };
        target
    } else if requested < (current >> 1).saturating_sub(5) {
        requested
    } else {
        return true;
    };

    resize_backing_to(obj, root_base, target)
}

/// The slots the list's current strategy has room for, or `None` for one that
/// carries no backing array at all.
///
/// # Safety
/// `list` must be a valid `W_ListObject`.
unsafe fn strategy_capacity(list: &W_ListObject) -> Option<usize> {
    Some(match list.strategy {
        ListStrategy::Object => list.object_items_capacity(),
        ListStrategy::Integer | ListStrategy::IntOrFloat => list.int_items.heap_capacity(),
        ListStrategy::Float => list.float_items.heap_capacity(),
        ListStrategy::Bytes => list.bytes_items.heap_capacity(),
        ListStrategy::Ascii => list.ascii_items.heap_capacity(),
        ListStrategy::Empty
        | ListStrategy::Size
        | ListStrategy::SimpleRange
        | ListStrategy::Range => return None,
    })
}

/// Publish a backing of exactly `target` slots for the list's current
/// strategy, reporting an allocation the machine cannot meet.
///
/// The shared tail of [`w_list_resize_hint`] and [`w_list_try_resize`]: those
/// two differ only in the capacity policy that picks `target`.
///
/// # Safety
/// `obj` must be a valid `W_ListObject` pinned at `root_base` with its guard
/// held, and its strategy must be one that carries a backing array.
unsafe fn resize_backing_to(obj: PyObjectRef, root_base: usize, target: usize) -> bool {
    let list = &mut *(obj as *mut W_ListObject);
    match list.strategy {
        ListStrategy::Object => W_ListObject::try_object_resize_capacity(obj, target).is_some(),
        ListStrategy::Integer | ListStrategy::IntOrFloat | ListStrategy::Float => {
            // The Integer and Float strategies share one `TypedItemsBlock`
            // resize and differ only in the field it lands in and the array
            // token it carries.
            let float = matches!(list.strategy, ListStrategy::Float);
            let (old, len, tid) = if float {
                (
                    list.float_items.block,
                    list.float_items.len(),
                    crate::object_array::gc_float_array_gc_type_id(),
                )
            } else {
                (
                    list.int_items.block,
                    list.int_items.len(),
                    crate::object_array::gc_int_array_gc_type_id(),
                )
            };
            let fresh = if target == 0 {
                crate::object_array::dealloc_typed_items_block(old);
                std::ptr::null_mut()
            } else {
                let Some(fresh) =
                    crate::object_array::try_grow_typed_items_block(old, target, len, tid)
                else {
                    return false;
                };
                fresh
            };
            let obj = crate::gc_roots::shadow_stack_get(root_base);
            let list = &mut *(obj as *mut W_ListObject);
            if float {
                list.float_items.block = fresh;
                if target == 0 {
                    list.float_items.set_len(0);
                }
            } else {
                list.int_items.block = fresh;
                if target == 0 {
                    list.int_items.set_len(0);
                }
            }
            true
        }
        ListStrategy::Bytes => W_ListObject::try_bytes_resize_capacity(obj, target).is_some(),
        ListStrategy::Ascii => W_ListObject::try_ascii_resize_capacity(obj, target).is_some(),
        ListStrategy::Empty
        | ListStrategy::Size
        | ListStrategy::SimpleRange
        | ListStrategy::Range => unreachable!(),
    }
}

/// CPython `list_resize` for a growth the caller must be able to refuse:
/// reserve the backing for `newsize` slots and report an allocation the
/// machine cannot meet, where the append path aborts.
///
/// `list_inplace_repeat_lock_held` sizes the whole result with one
/// `list_resize` before it copies anything, and `ll_inplace_mul` the same with
/// one `_ll_resize`; both surface a refusal as `MemoryError`, so the count is
/// never walked as a loop trip.
///
/// Returns false only for that refusal.  A `newsize` the block already covers
/// is `list_resize`'s "failure is impossible if newsize <= self.allocated"
/// case and answers true without touching the array.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_try_resize(obj: PyObjectRef, newsize: usize) -> bool {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    let obj = crate::gc_roots::pin_root(obj);
    let _list_guard = w_list_lock(obj);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let list = &mut *(obj as *mut W_ListObject);
    // A strategy with no backing array has nothing to size; the range ones
    // inherit `BaseRangeListStrategy._resize_hint`'s documented no-op.
    let Some(current) = strategy_capacity(list) else {
        return true;
    };
    let old_size = list.live_len();
    let target = list.resized_allocation(old_size, newsize);
    if target > current && !resize_backing_to(obj, root_base, target) {
        return false;
    }
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    (*(obj as *mut W_ListObject)).allocated = target as isize;
    true
}

/// The capacity `list.extend` reserves for `extra` more items, or `None` where
/// the sum it would need does not fit and the reservation is skipped.
///
/// # Safety
/// `list` must be a valid `W_ListObject` with its guard held.
unsafe fn reserve_for_extend_target(list: &W_ListObject, extra: usize) -> Option<usize> {
    if list.allocated == 0 {
        return Some((extra + 1) & !1);
    }
    let old_size = list.live_len();
    old_size
        .checked_add(extra)
        .map(|new_size| list.resized_allocation(old_size, new_size))
}

/// Reserve CPython's logical slots before `list.extend` consumes its source.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_reserve_for_extend(obj: PyObjectRef, extra: usize) {
    if extra == 0 {
        return;
    }
    let _list_guard = w_list_lock(obj);
    let list = &mut *(obj as *mut W_ListObject);
    if let Some(target) = reserve_for_extend_target(list, extra) {
        list.allocated = target as isize;
    }
}

/// [`w_list_reserve_for_extend`] for an `extra` the source chose: size the
/// backing as well, and report an allocation the machine cannot meet.
///
/// `list_extend_iter_lock_held` reserves with the ordinary `list_resize`,
/// whose failure the extend reports as `MemoryError`.  A `__length_hint__` is
/// the one input to that reservation a Python object picks, so it is the one
/// that can name a length no allocation can serve.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_try_reserve_for_extend(obj: PyObjectRef, extra: usize) -> bool {
    if extra == 0 {
        return true;
    }
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    let obj = crate::gc_roots::pin_root(obj);
    let _list_guard = w_list_lock(obj);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let list = &mut *(obj as *mut W_ListObject);
    let Some(target) = reserve_for_extend_target(list, extra) else {
        return true;
    };
    // A strategy with no backing array has nothing to size, and the append
    // that follows is what moves it off that strategy.
    if let Some(current) = strategy_capacity(list)
        && target > current
        && !resize_backing_to(obj, root_base, target)
    {
        return false;
    }
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    (*(obj as *mut W_ListObject)).allocated = target as isize;
    true
}

/// `list_extend_set` / `list_extend_dict` use ordinary `list_resize` even
/// when the destination has no backing array; unlike sequence-fast extension
/// they do not call `list_preallocate_exact`.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_resize_for_extend(obj: PyObjectRef, extra: usize) {
    if extra == 0 {
        return;
    }
    let _list_guard = w_list_lock(obj);
    let list = &mut *(obj as *mut W_ListObject);
    let old_size = list.live_len();
    if let Some(new_size) = old_size.checked_add(extra) {
        list.allocated = list.resized_allocation(old_size, new_size) as isize;
    }
}

/// `list_extend_iter_lock_held` trims an overestimated length hint after the
/// iterator ends, using ordinary `list_resize` shrink rules.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_finish_extend(obj: PyObjectRef) {
    let _list_guard = w_list_lock(obj);
    let list = &mut *(obj as *mut W_ListObject);
    let size = list.live_len();
    if list.allocated > size as isize {
        list.allocated = list.resized_allocation(size, size) as isize;
    }
}

/// Recompute one CPython `list_resize` after a pyre implementation performed
/// a batch mutation as several primitive removals.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_finish_batch_resize(obj: PyObjectRef, old_size: usize, old_allocated: isize) {
    let _list_guard = w_list_lock(obj);
    let list = &mut *(obj as *mut W_ListObject);
    list.allocated = old_allocated;
    list.sync_allocated(old_size);
}

/// Set CPython's raw `PyListObject.allocated` field. `list.sort` uses `-1`
/// while the saved item array is detached, then restores the previous value.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_set_allocated(obj: PyObjectRef, allocated: isize) {
    let _list_guard = w_list_lock(obj);
    (*(obj as *mut W_ListObject)).allocated = allocated;
}

/// Whether `obj` is a list currently backed by the Integer strategy — the
/// only shape [`w_list_int_set_len`] can rewind.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_is_integer_strategy(obj: PyObjectRef) -> bool {
    (*(obj as *const W_ListObject)).strategy == ListStrategy::Integer
}

/// Check whether appending one element can complete without reallocating.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_can_append_without_realloc(obj: PyObjectRef) -> bool {
    let list = &*(obj as *const W_ListObject);
    match list.strategy {
        // EmptyListStrategy holds no array yet — first append always reallocates.
        ListStrategy::Empty | ListStrategy::Size => false,
        ListStrategy::SimpleRange | ListStrategy::Range => false,
        ListStrategy::Object => list.object_spare_capacity() > 0,
        ListStrategy::Integer => list.int_items.spare_capacity() > 0,
        ListStrategy::IntOrFloat => list.int_items.spare_capacity() > 0,
        ListStrategy::Float => list.float_items.spare_capacity() > 0,
        ListStrategy::Bytes => list.bytes_items.spare_capacity() > 0,
        ListStrategy::Ascii => list.ascii_items.spare_capacity() > 0,
    }
}

/// Check whether the list is currently using inline array storage.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_is_inline_storage(obj: PyObjectRef) -> bool {
    let list = &*(obj as *const W_ListObject);
    match list.strategy {
        // EmptyListStrategy.lstorage = self.erase(None) — no backing array.
        ListStrategy::Empty | ListStrategy::Size => false,
        ListStrategy::SimpleRange | ListStrategy::Range => false,
        // Object strategy stores items in a GC-shaped `ItemsBlock`, never
        // an inline allocation — upstream rlist.py doesn't have an
        // "inline" bit either.
        ListStrategy::Object => false,
        ListStrategy::Integer => list.int_items.is_inline(),
        ListStrategy::IntOrFloat => list.int_items.is_inline(),
        ListStrategy::Float => list.float_items.is_inline(),
        ListStrategy::Bytes => list.bytes_items.is_inline(),
        ListStrategy::Ascii => list.ascii_items.is_inline(),
    }
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_uses_object_storage(obj: PyObjectRef) -> bool {
    let list = &*(obj as *const W_ListObject);
    list.strategy == ListStrategy::Object
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_uses_int_storage(obj: PyObjectRef) -> bool {
    let list = &*(obj as *const W_ListObject);
    list.strategy == ListStrategy::Integer
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_uses_float_storage(obj: PyObjectRef) -> bool {
    let list = &*(obj as *const W_ListObject);
    list.strategy == ListStrategy::Float
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_uses_ascii_storage(obj: PyObjectRef) -> bool {
    let list = &*(obj as *const W_ListObject);
    list.strategy == ListStrategy::Ascii
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_uses_int_or_float_storage(obj: PyObjectRef) -> bool {
    let list = &*(obj as *const W_ListObject);
    list.strategy == ListStrategy::IntOrFloat
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_uses_empty_storage(obj: PyObjectRef) -> bool {
    let list = &*(obj as *const W_ListObject);
    list.strategy == ListStrategy::Empty
}

/// True when the next `w_list_append` on `obj` takes the Object strategy's
/// in-place arm (`rlist.py:285` resize-ge fast case) into a GC-managed items
/// block: spare capacity, so the store is an item-slot write and the list's
/// `items` pointer does not change.
///
/// The tracer uses this to decide whether the recorded trace needs
/// `list_write_barrier` at all — see `FbwWalkMode::append_inplace_wb_covered`.
/// A `std::alloc` block (`MAJIT_GC_ITEMSBLOCK=0`) answers false: it has no GC
/// header for an array barrier to mark, so the barrier on the enclosing
/// `W_ListObject` is the only thing keeping the block's slots reachable.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_append_stores_into_gc_block_in_place(obj: PyObjectRef) -> bool {
    let list = &*(obj as *const W_ListObject);
    list.strategy == ListStrategy::Object
        && ll_list_obj_length(list) < ll_list_obj_capacity(list)
        && !list.items.is_null()
        && crate::gc_hook::try_gc_owns_object(list.items as *mut u8)
}

/// Rebuild the list's object storage from a Vec.
unsafe fn rebuild_object_items(list: &mut W_ListObject, items: Vec<PyObjectRef>) {
    list.set_object_items_from_vec(items);
}

/// Snapshot all items of a list as a `Vec<PyObjectRef>`, regardless of
/// strategy. Integer/Float items are wrapped into `W_IntObject` /
/// `W_FloatObject`, and Bytes items are re-wrapped from their erased strings,
/// matching listobject.py `AbstractUnwrappedStrategy.getitems_copy` and
/// `_temporarily_as_objects()`. Used by callers outside `pyre-object`
/// (e.g. the interpreter's unpack / set-update / list-to-tuple paths)
/// that need a uniform object view.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_items_copy_as_vec(obj: PyObjectRef) -> Vec<PyObjectRef> {
    w_list_items_copy_as_vec_mode(obj, false)
}

/// JIT-context form of [`w_list_items_copy_as_vec`].
///
/// `AbstractUnwrappedStrategy.getitems_copy` deliberately skips its
/// consecutive-wrapper reuse while `jit.we_are_jitted()`: identity-bearing
/// boxes stay explicit in the trace. The interpreter crate supplies that
/// symbolic boolean because `pyre-object` sits below `majit-metainterp`.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_items_copy_as_vec_mode(
    obj: PyObjectRef,
    we_are_jitted: bool,
) -> Vec<PyObjectRef> {
    let list = &*(obj as *const W_ListObject);
    temporarily_as_objects(list, we_are_jitted)
}

/// Raw `(ptr, len)` view of an Object-strategy list's `PyObjectRef` items for
/// GC root walking. Returns `None` for Empty / Integer / Float / Bytes
/// strategies: scalar strategies have no GC children, while Bytes storage is
/// walked by `list_object_custom_trace`; materialising either representation
/// would allocate — forbidden while the collector is marking.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.  The returned pointer aliases
/// the list's live backing store; the caller must not mutate the list while
/// reading through it.
pub unsafe fn w_list_object_items_ptr_len(obj: PyObjectRef) -> Option<(*const PyObjectRef, usize)> {
    let list = &*(obj as *const W_ListObject);
    match list.strategy {
        ListStrategy::Object => Some((items_block_items_base(list.items), list.length_relaxed())),
        _ => None,
    }
}

/// listobject.py _temporarily_as_objects()
///
/// Returns wrapped object items without mutating the source list's strategy.
/// PyPy creates a temporary W_ListObject with ObjectListStrategy; Rust
/// returns a Vec<PyObjectRef> copy instead.
unsafe fn temporarily_as_objects(list: &W_ListObject, we_are_jitted: bool) -> Vec<PyObjectRef> {
    match list.strategy {
        // listobject.py EmptyListStrategy.getitems returns [].
        ListStrategy::Empty | ListStrategy::Size => Vec::new(),
        ListStrategy::SimpleRange | ListStrategy::Range => {
            let values = range_list_values(list);
            boxed_from_ints(&values, we_are_jitted)
        }
        ListStrategy::Object => list.object_to_vec(),
        ListStrategy::Integer => {
            // Copy scalar storage before wrapping: allocation may move the
            // typed block, while getitems_copy itself works from its stable
            // unwrapped list snapshot.
            let values = list.int_items.as_slice().to_vec();
            boxed_from_ints(&values, we_are_jitted)
        }
        ListStrategy::IntOrFloat => {
            let values = list.int_items.as_slice().to_vec();
            boxed_from_int_or_float(&values, we_are_jitted)
        }
        ListStrategy::Float => {
            let values = list.float_items.as_slice().to_vec();
            boxed_from_floats(&values, we_are_jitted)
        }
        ListStrategy::Bytes => {
            // The wraps allocate, so the list has to be reachable by slot for
            // the re-read `boxed_from_bytes` does per element.
            let _roots = crate::gc_roots::push_roots();
            let _ = crate::gc_roots::pin_root(
                (list as *const W_ListObject as *mut W_ListObject) as PyObjectRef,
            );
            let obj_slot = crate::gc_roots::shadow_stack_len() - 1;
            boxed_from_bytes(obj_slot, we_are_jitted)
        }
        ListStrategy::Ascii => {
            let _roots = crate::gc_roots::push_roots();
            let _ = crate::gc_roots::pin_root(
                (list as *const W_ListObject as *mut W_ListObject) as PyObjectRef,
            );
            let obj_slot = crate::gc_roots::shadow_stack_len() - 1;
            boxed_from_ascii(obj_slot, we_are_jitted)
        }
    }
}

fn normalize_insert_index(index: i64, len: usize) -> usize {
    if index < 0 {
        (index + len as i64).max(0) as usize
    } else {
        (index as usize).min(len)
    }
}

/// listobject.py IntegerListStrategy.insert
/// Strategy-preserving: inserts on typed storage when type matches,
/// switches to Object only when incompatible.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_insert(obj: PyObjectRef, index: i64, value: PyObjectRef) {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::publish_roots(&[obj, value]);
    crate::gc_roots::normalize_roots(root_base, 2);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let value = crate::gc_roots::shadow_stack_get(root_base + 1);
    let list = &mut *(obj as *mut W_ListObject);
    let old_size = list.live_len();
    match list.strategy {
        // EmptyListStrategy doesn't override insert, so it falls through
        // ListStrategy.insert (listobject.py) → switches to typed strategy
        // via append. Mirror by switching first then re-dispatching.
        ListStrategy::Empty | ListStrategy::Size => {
            switch_to_correct_strategy(list, value);
            w_list_insert(
                crate::gc_roots::shadow_stack_get(root_base),
                index,
                crate::gc_roots::shadow_stack_get(root_base + 1),
            );
        }
        ListStrategy::SimpleRange | ListStrategy::Range => {
            let obj = switch_range_to_integer_strategy(list);
            w_list_insert(obj, index, crate::gc_roots::shadow_stack_get(root_base + 1));
        }
        ListStrategy::Integer => {
            if is_plain_int1(value) {
                let idx = normalize_insert_index(index, list.int_items.len());
                list.int_items.insert(idx, plain_int_w(value));
                list.sync_allocated(old_size);
                return;
            }
            if is_float_strategy_item(value) && integer_to_int_or_float(list) {
                w_list_insert(
                    crate::gc_roots::shadow_stack_get(root_base),
                    index,
                    crate::gc_roots::shadow_stack_get(root_base + 1),
                );
            } else {
                switch_to_object_strategy(list);
                w_list_insert(
                    crate::gc_roots::shadow_stack_get(root_base),
                    index,
                    crate::gc_roots::shadow_stack_get(root_base + 1),
                );
            }
        }
        ListStrategy::IntOrFloat => {
            if let Some(value) = int_or_float_encode_item(value) {
                let idx = normalize_insert_index(index, list.int_items.len());
                list.int_items.insert(idx, value);
                list.sync_allocated(old_size);
                return;
            }
            switch_to_object_strategy(list);
            w_list_insert(
                crate::gc_roots::shadow_stack_get(root_base),
                index,
                crate::gc_roots::shadow_stack_get(root_base + 1),
            );
        }
        ListStrategy::Float => {
            if is_float_strategy_item(value) {
                let idx = normalize_insert_index(index, list.float_items.len());
                list.float_items.insert(idx, w_float_get_value(value));
                list.sync_allocated(old_size);
                return;
            }
            if is_plain_int1(value)
                && int_or_float_encode_int(plain_int_w(value)).is_some()
                && float_to_int_or_float(list)
            {
                w_list_insert(
                    crate::gc_roots::shadow_stack_get(root_base),
                    index,
                    crate::gc_roots::shadow_stack_get(root_base + 1),
                );
            } else {
                switch_to_object_strategy(list);
                w_list_insert(
                    crate::gc_roots::shadow_stack_get(root_base),
                    index,
                    crate::gc_roots::shadow_stack_get(root_base + 1),
                );
            }
        }
        ListStrategy::Object => {
            let idx = normalize_insert_index(index, list.length_relaxed());
            list.object_insert(idx, value);
            let obj = crate::gc_roots::shadow_stack_get(root_base);
            list_write_barrier(obj);
            let obj = crate::gc_roots::shadow_stack_get(root_base);
            let list = &mut *(obj as *mut W_ListObject);
            list.sync_allocated(old_size);
        }
        ListStrategy::Bytes => {
            if is_bytes_strategy_item(value) {
                let idx = normalize_insert_index(index, list.bytes_items.len());
                let value = prepare_list_ref_store(obj, value);
                let obj = current_gc_ref(obj);
                let list = &*(obj as *const W_ListObject);
                // Same reservation the append arm makes: `insert` may not
                // publish a fresh block itself.
                let value = if list.bytes_items.spare_capacity() == 0 {
                    w_list_grow_bytes_block(obj, value)
                } else {
                    value
                };
                let obj = current_gc_ref(obj);
                let list = &mut *(obj as *mut W_ListObject);
                list.bytes_items.insert(idx, w_bytes_block(value));
                list.sync_allocated(old_size);
            } else {
                switch_to_object_strategy(list);
                w_list_insert(
                    crate::gc_roots::shadow_stack_get(root_base),
                    index,
                    crate::gc_roots::shadow_stack_get(root_base + 1),
                );
            }
        }
        ListStrategy::Ascii => {
            if is_ascii_strategy_item(value) {
                let idx = normalize_insert_index(index, list.ascii_items.len());
                let value = prepare_list_ref_store(obj, value);
                let obj = current_gc_ref(obj);
                let list = &*(obj as *const W_ListObject);
                let value = if list.ascii_items.spare_capacity() == 0 {
                    w_list_grow_ascii_block(obj, value)
                } else {
                    value
                };
                let obj = current_gc_ref(obj);
                let list = &mut *(obj as *mut W_ListObject);
                list.ascii_items
                    .insert(idx, w_str_storage(value) as *const _);
                list.sync_allocated(old_size);
            } else {
                switch_to_object_strategy(list);
                w_list_insert(
                    crate::gc_roots::shadow_stack_get(root_base),
                    index,
                    crate::gc_roots::shadow_stack_get(root_base + 1),
                );
            }
        }
    }
}

/// listobject.py IntegerListStrategy.pop
/// Strategy-preserving: pops from typed storage, wraps result.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_pop(obj: PyObjectRef, index: i64) -> Option<PyObjectRef> {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    let obj = crate::gc_roots::pin_root(obj);
    let _list_guard = w_list_lock(obj);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let list = &mut *(obj as *mut W_ListObject);
    let old_size = list.live_len();
    let result = match list.strategy {
        // listobject.py EmptyListStrategy.pop raises IndexError.
        ListStrategy::Empty | ListStrategy::Size => None,
        ListStrategy::SimpleRange => {
            // SimpleRangeListStrategy.pop always materialises; only the
            // separate pop_end hook preserves the strategy.
            let obj = switch_range_to_integer_strategy(list);
            return w_list_pop(obj, index);
        }
        ListStrategy::Range => {
            let len = range_list_length(list) as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return None;
            }
            let (start, step) = range_list_start_step(list);
            if idx == 0 {
                let result = w_int_new(start);
                let result_slot = crate::gc_roots::shadow_stack_len();
                let _ = crate::gc_roots::pin_root(result);
                let obj = crate::gc_roots::shadow_stack_get(root_base);
                let _ =
                    install_range_state(obj, ListStrategy::Range, &[start + step, step, len - 1]);
                Some(crate::gc_roots::shadow_stack_get(result_slot))
            } else if idx == len - 1 {
                let result = w_int_new(start + idx * step);
                let result_slot = crate::gc_roots::shadow_stack_len();
                let _ = crate::gc_roots::pin_root(result);
                let obj = crate::gc_roots::shadow_stack_get(root_base);
                let _ = install_range_state(obj, ListStrategy::Range, &[start, step, len - 1]);
                Some(crate::gc_roots::shadow_stack_get(result_slot))
            } else {
                let obj = switch_range_to_integer_strategy(list);
                return w_list_pop(obj, index);
            }
        }
        ListStrategy::Integer => {
            let len = list.int_items.len() as i64;
            if len == 0 {
                return None;
            }
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return None;
            }
            let item = list.int_items.remove(idx as usize);
            Some(w_int_new(item))
        }
        ListStrategy::IntOrFloat => {
            let len = list.int_items.len() as i64;
            if len == 0 {
                return None;
            }
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return None;
            }
            let item = list.int_items.remove(idx as usize);
            Some(if int_or_float_is_int(item) {
                w_int_new(int_or_float_decode_int(item))
            } else {
                w_float_new(f64::from_bits(item as u64))
            })
        }
        ListStrategy::Float => {
            let len = list.float_items.len() as i64;
            if len == 0 {
                return None;
            }
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return None;
            }
            let item = list.float_items.remove(idx as usize);
            Some(w_float_new(item))
        }
        ListStrategy::Object => {
            let len = list.length_relaxed() as i64;
            if len == 0 {
                return None;
            }
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return None;
            }
            Some(list.object_remove(idx as usize))
        }
        ListStrategy::Bytes => {
            let len = list.bytes_items.len() as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return None;
            }
            Some(w_bytes_from_block(list.bytes_items.remove(idx as usize)))
        }
        ListStrategy::Ascii => {
            let len = list.ascii_items.len() as i64;
            let idx = if index < 0 { index + len } else { index };
            if idx < 0 || idx >= len {
                return None;
            }
            Some(w_str_from_storage(
                list.ascii_items.remove(idx as usize) as *mut _
            ))
        }
    };
    if result.is_some() {
        // The object-strategy remove runs the before-move barrier, whose
        // entry is a safepoint.  Reload the list the outer bracket kept live
        // before updating its accounting fields.
        let list = &mut *(crate::gc_roots::shadow_stack_get(root_base) as *mut W_ListObject);
        list.sync_allocated(old_size);
    }
    result
}

/// Remove and return the last item. Returns `None` if empty.
///
/// Splits into a guard-taking wrapper and a lock-free
/// [`w_list_pop_end_inner`], matching [`w_list_append`] /
/// [`w_list_append_inner`], because the pop fold descends this body:
///
/// * the wrapper must stay look-inside — the codewriter only reaches graphs
///   through look-inside calls from a jitdriver portal
///   (`grab_initial_jitcodes` / `enum_pending_graphs`), so a
///   `dont_look_inside` wrapper hides the inner body from the pipeline as
///   well;
/// * the descended body must hold no guard — a `w_list_lock` acquire/release
///   pair inside it declines the fold's sub-walk.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_pop_end(obj: PyObjectRef) -> Option<PyObjectRef> {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    let obj = crate::gc_roots::pin_root(obj);
    let _list_guard = w_list_lock(obj);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let list = &mut *(obj as *mut W_ListObject);
    let length = list.live_len();
    // Keep the empty check in the descended body as well as the mutation:
    // listobject.py W_ListObject.descr_pop checks length before pop_end.
    let w_item = w_list_pop_end_inner(obj)?;
    // The inner body can allocate. Reload the rooted list before accounting.
    let list = &mut *(crate::gc_roots::shadow_stack_get(root_base) as *mut W_ListObject);
    list.sync_allocated(length);
    Some(w_item)
}

/// [`w_list_pop_end`]'s checked body, run with the list's guard already held.
///
/// The generated pop descent must include `W_ListObject.descr_pop`'s empty
/// check, not just `AbstractUnwrappedStrategy.pop_end`'s unchecked mutation.
/// `None` is the existing low-level error result consumed by list_method_pop;
/// a guard on the non-empty branch resumes that caller before the pop.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject` and the caller must hold
/// `w_list_lock(obj)`.
pub unsafe fn w_list_pop_end_inner(obj: PyObjectRef) -> Option<PyObjectRef> {
    let list = &mut *(obj as *mut W_ListObject);
    let length = match list.strategy {
        ListStrategy::Empty | ListStrategy::Size => 0,
        ListStrategy::SimpleRange | ListStrategy::Range => range_list_length(list),
        ListStrategy::Integer => ll_list_int_length(list),
        ListStrategy::IntOrFloat => list.int_items.len(),
        ListStrategy::Float => list.float_items.len(),
        ListStrategy::Object => list.length_relaxed(),
        ListStrategy::Bytes => list.bytes_items.len(),
        ListStrategy::Ascii => list.ascii_items.len(),
    };
    if length == 0 {
        return None;
    }
    Some(match list.strategy {
        // EmptyListStrategy.pop is unreachable after descr_pop's length check
        // (pypy/objspace/std/listobject.py).
        ListStrategy::Empty | ListStrategy::Size => PY_NULL,
        ListStrategy::SimpleRange => {
            let length = range_list_length(list);
            let result = w_int_new((length - 1) as i64);
            let result_slot = crate::gc_roots::shadow_stack_len();
            let _ = crate::gc_roots::pin_root(result);
            if length > 1 {
                let obj = current_gc_ref(obj);
                let _ = install_range_state(obj, ListStrategy::SimpleRange, &[(length - 1) as i64]);
            } else {
                let obj = current_gc_ref(obj);
                let list = &mut *(obj as *mut W_ListObject);
                list.items = std::ptr::null_mut();
                list.strategy = ListStrategy::Empty;
            }
            crate::gc_roots::shadow_stack_get(result_slot)
        }
        ListStrategy::Range => {
            let length = range_list_length(list);
            let (start, step) = range_list_start_step(list);
            let result = w_int_new(start + ((length - 1) as i64) * step);
            let result_slot = crate::gc_roots::shadow_stack_len();
            let _ = crate::gc_roots::pin_root(result);
            let obj = current_gc_ref(obj);
            let _ = install_range_state(
                obj,
                ListStrategy::Range,
                &[start, step, (length - 1) as i64],
            );
            crate::gc_roots::shadow_stack_get(result_slot)
        }
        ListStrategy::Integer => {
            let length = ll_list_int_length(list);
            // rpython/rtyper/rlist.py ll_pop_default's internal precondition,
            // separate from the checked empty branch above.
            assert!(length > 0, "pop from empty list");
            let index = length - 1;
            let item = ll_list_int_getitem_fast(list, index);
            ll_list_int_set_len(list, index);
            w_int_new(item)
        }
        ListStrategy::IntOrFloat => {
            let item = list.int_items.pop();
            if int_or_float_is_int(item) {
                w_int_new(int_or_float_decode_int(item))
            } else {
                w_float_new(f64::from_bits(item as u64))
            }
        }
        ListStrategy::Float => w_float_new(list.float_items.pop()),
        // `AbstractUnwrappedStrategy.pop_end` (listobject.py) under an identity
        // `wrap`, rtyped as `ll_pop_default` (rlist.py): read, null the vacated
        // GC slot, shrink. Decomposed through the `ll_list_obj_*` leaves, like
        // the Integer arm, so the orthodox pop fold can descend it.
        ListStrategy::Object => {
            let length = ll_list_obj_length(list);
            // rpython/rtyper/rlist.py ll_pop_default's internal precondition.
            assert!(length > 0, "pop from empty list");
            let index = length - 1;
            let item = ll_list_obj_getitem_fast(list, index);
            ll_list_obj_setitem_fast(list, index, PY_NULL);
            ll_list_obj_set_len(list, index);
            item
        }
        ListStrategy::Bytes => w_bytes_from_block(list.bytes_items.pop()),
        ListStrategy::Ascii => w_str_from_storage(list.ascii_items.pop() as *mut _),
    })
}

/// The unwrapped Integer-strategy storage `IntegerListStrategy.sort`
/// (listobject.py:1963) orders in place, or `None` when the list holds a
/// different strategy.  Handed out as a raw pointer + length because the
/// caller (the `descr_sort` level) owns the sort; nothing in the sort boxes a
/// value, so no collection can move the block while it is ordered.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`, and the returned pointer is
/// only valid until the list's storage is next resized or re-strategised.
pub unsafe fn w_list_int_items_raw(obj: PyObjectRef) -> Option<(*mut i64, usize)> {
    let list = &mut *(obj as *mut W_ListObject);
    if list.strategy != ListStrategy::Integer {
        return None;
    }
    let items = list.int_items.as_mut_slice();
    Some((items.as_mut_ptr(), items.len()))
}

/// `BaseRangeListStrategy.sort`: an arithmetic progression already ordered in
/// the requested direction keeps its compact storage.  The opposite direction
/// first becomes `IntegerListStrategy`, after which the ordinary scalar sorter
/// must run (reported by returning `false`).
pub unsafe fn w_list_sort_range(obj: PyObjectRef, reverse: bool) -> bool {
    let list = &mut *(obj as *mut W_ListObject);
    if !matches!(
        list.strategy,
        ListStrategy::SimpleRange | ListStrategy::Range
    ) {
        return false;
    }
    let (_, step) = range_list_start_step(list);
    if (step > 0 && reverse) || (step < 0 && !reverse) {
        let _ = switch_range_to_integer_strategy(list);
        false
    } else {
        true
    }
}

/// The Float-strategy counterpart of [`w_list_int_items_raw`]
/// (`FloatListStrategy.sort`, listobject.py).
///
/// # Safety
/// As [`w_list_int_items_raw`].
pub unsafe fn w_list_float_items_raw(obj: PyObjectRef) -> Option<(*mut f64, usize)> {
    let list = &mut *(obj as *mut W_ListObject);
    if list.strategy != ListStrategy::Float {
        return None;
    }
    let items = list.float_items.as_mut_slice();
    Some((items.as_mut_ptr(), items.len()))
}

/// listobject.py IntOrFloatListStrategy.sort and
/// listobject.py IntOrFloatSort.lt.  Unlike the homogeneous raw-array
/// accessors above, the encoded `i64` values must be ordered after decoding.
/// Reverse follows PyPy's reverse/stable-sort/reverse sequence so equal
/// int/float values retain reverse-sort stability.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_sort_int_or_float(obj: PyObjectRef, reverse: bool) -> bool {
    let list = &mut *(obj as *mut W_ListObject);
    if list.strategy != ListStrategy::IntOrFloat {
        return false;
    }
    let items = list.int_items.as_mut_slice();
    if reverse {
        items.reverse();
    }
    items.sort_by(|a, b| {
        int_or_float_as_float(*a)
            .partial_cmp(&int_or_float_as_float(*b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if reverse {
        items.reverse();
    }
    true
}

/// `BytesListStrategy.sort` / `AsciiListStrategy.sort` (`listobject.py`):
/// order the erased RPython string payloads directly with `StringSort`, then
/// apply the upstream reverse step without allocating Python wrappers.
pub unsafe fn w_list_sort_strings(obj: PyObjectRef, reverse: bool) -> bool {
    let list = &mut *(obj as *mut W_ListObject);
    match list.strategy {
        ListStrategy::Bytes => list.bytes_items.as_mut_slice().sort_by(|a, b| {
            crate::bytesobject::bytes_block_chars(*a).cmp(crate::bytesobject::bytes_block_chars(*b))
        }),
        ListStrategy::Ascii => list
            .ascii_items
            .as_mut_slice()
            .sort_by(|a, b| (&**a).as_bytes().cmp((&**b).as_bytes())),
        _ => return false,
    }
    if reverse {
        match list.strategy {
            ListStrategy::Bytes => list.bytes_items.reverse(),
            ListStrategy::Ascii => list.ascii_items.reverse(),
            _ => unreachable!(),
        }
    }
    true
}

/// Whether the list still holds the EmptyListStrategy.
///
/// `descr_sort` (listobject.py) uses this to tell whether the user mucked
/// with the receiver while it was emptied for the sort: any mutation switches
/// the list off the Empty strategy and a list never switches back, so an
/// append followed by a pop is caught even though the length is 0 again.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`.
pub unsafe fn w_list_is_empty_strategy(obj: PyObjectRef) -> bool {
    (*(obj as *const W_ListObject)).strategy == ListStrategy::Empty
}

/// Whether `obj` uses either compact `BaseRangeListStrategy` storage shape.
pub unsafe fn w_list_is_range_strategy(obj: PyObjectRef) -> bool {
    matches!(
        (*(obj as *const W_ListObject)).strategy,
        ListStrategy::SimpleRange | ListStrategy::Range
    )
}

/// SizeListStrategy.get_sizehint; `None` for every other strategy.
pub unsafe fn w_list_sizehint(obj: PyObjectRef) -> Option<i64> {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    let obj = crate::gc_roots::pin_root(obj);
    let _list_guard = w_list_lock(obj);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let list = &*(obj as *const W_ListObject);
    (list.strategy == ListStrategy::Size).then(|| sizehint_state_value(list.items))
}

/// listobject.py `W_ListObject.__init__` applied to an existing list:
/// re-pick the strategy for `items` and install fresh storage, dropping
/// whatever the list held.  This is how `descr_sort` (listobject.py) puts
/// the sorted items back — one bulk install, not an append loop, so the
/// storage is sized once and the strategy is decided from the whole item set.
///
/// # Safety
/// `obj` must point to a valid `W_ListObject`; every element of `items` live.
pub unsafe fn w_list_init_items(obj: PyObjectRef, items: Vec<PyObjectRef>) {
    // Build the replacement storage before touching the list: the allocations
    // below can collect, and `items` is only reachable through the caller's
    // roots until the install.
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    let strategy = list_strategy_for(&items);
    let mut storage = build_list_storage(&items, strategy);
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    // `w_list_clear` is the twin destructive re-installation and holds the
    // stripe lock across its `drop_object_items`; a concurrent reader must not
    // see the half-installed state this one writes either.  Take the lock after
    // the allocating `build_list_storage` so the acquire cannot deadlock behind
    // a collection, and reload `obj` behind it because a contended acquire
    // blocks through `before_external_block`.
    // The Object items block needs the same bracket as `obj`, and for the same
    // reason: `build_list_storage` hands it back young, with no shadow-stack
    // slot (`alloc_list_items_block_gc`'s `push_roots` scope ends at its
    // return) and no heap edge until the store below, so a collection that runs
    // while this thread sits in `before_external_block` sees it as garbage.
    // `reload_typed_blocks` covers only the typed blocks -- `ListStorage` keeps
    // a root slot for those two alone.  Same pin/reload
    // `w_list_new_with_strategy` puts around its header allocation.
    let block_root: Option<usize> = if storage.block.is_null() {
        None
    } else {
        let s = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(storage.block as PyObjectRef);
        Some(s)
    };
    let _list_guard = w_list_lock(obj);
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let list = &mut *(obj as *mut W_ListObject);
    // `drop_object_items`' `try_gc_owns_object` query is a safepoint and the
    // fresh blocks have no heap edge until the stores below, so close their pin
    // bracket only once it is behind them (`IntArray::install`).
    if list.strategy == ListStrategy::Object {
        list.drop_object_items();
    } else {
        list.items = std::ptr::null_mut();
    }
    storage.reload_typed_blocks();
    if let Some(s) = block_root {
        storage.block = crate::gc_roots::shadow_stack_get(s) as *mut ItemsBlock;
    }
    list.set_length_relaxed(storage.length);
    list.items = storage.block;
    list.strategy = strategy;
    list.int_items = storage.int_items;
    list.float_items = storage.float_items;
    list.bytes_items = storage.bytes_items;
    list.ascii_items = storage.ascii_items;
    // Object and Bytes storage both publish a freshly allocated GC block from
    // an existing list header. Integer/Float blocks are old-generation leaf
    // arrays and need no remembered-set edge.
    if matches!(
        strategy,
        ListStrategy::Object | ListStrategy::Bytes | ListStrategy::Ascii
    ) {
        list_write_barrier(obj);
    }
}

/// listobject.py W_ListObject.clear — switches to EmptyListStrategy.
///
/// Drops any typed storage and resets the list to the EmptyListStrategy
/// state, exactly like PyPy. The next append will pick a fresh typed
/// strategy via switch_to_correct_strategy.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_clear(obj: PyObjectRef) {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    let obj = crate::gc_roots::pin_root(obj);
    let _list_guard = w_list_lock(obj);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let list = &mut *(obj as *mut W_ListObject);
    // `list_sort_impl` has already detached the storage and set size to zero.
    // CPython's `list.clear()` is then a no-op and leaves the -1 sentinel, so
    // it must not spuriously report "list modified during sort".
    if list.live_len() == 0 && list.allocated == -1 {
        return;
    }
    if list.strategy == ListStrategy::Object {
        list.drop_object_items();
    } else {
        list.items = std::ptr::null_mut();
        list.set_length_relaxed(0);
    }
    // Empty strategy reads neither typed array; the next append reinstalls the
    // matching one through `switch_to_correct_strategy`.
    list.int_items.install(IntArray::empty());
    list.float_items.install(FloatArray::empty());
    list.bytes_items.install(BytesArray::empty());
    list.ascii_items.install(UnicodeArray::empty());
    list.strategy = ListStrategy::Empty;
    list.set_length_relaxed(0);
    list.allocated = 0;
}

/// listobject.py EmptyListStrategy.switch_to_correct_strategy —
/// public entry for the JIT's empty-append staging. It selects the strategy and
/// applies the first `_ll_list_resize_ge` growth (0 -> 4) WITHOUT changing the
/// logical length or storing the item. The trace emitter stages the same array;
/// the caller then performs the append through the spare-capacity leg. Only
/// valid on an Empty-strategy list.
/// # Safety
/// `obj` must point to a valid Empty-strategy `W_ListObject`; `value` live.
pub unsafe fn w_list_switch_to_strategy_for(obj: PyObjectRef, value: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::publish_roots(&[obj, value]);
    crate::gc_roots::normalize_roots(root_base, 2);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let value = crate::gc_roots::shadow_stack_get(root_base + 1);
    let list = &mut *(obj as *mut W_ListObject);
    debug_assert_eq!(list.strategy, ListStrategy::Empty);
    let _ = switch_to_correct_strategy(list, value);
    // `switch_to_correct_strategy` owns a nested root scope. Reload the outer
    // slot before the first resize, then follow rlist's `_ll_list_resize_ge`
    // allocation discipline: retain the old block only as an allocation
    // input, and reload the possibly moved list before publishing the result.
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let list = &mut *(obj as *mut W_ListObject);
    match list.strategy {
        ListStrategy::Integer => {
            let fresh = crate::object_array::grow_typed_items_block(
                list.int_items.block,
                4,
                0,
                crate::object_array::gc_int_array_gc_type_id(),
            );
            let obj = crate::gc_roots::shadow_stack_get(root_base);
            (*(obj as *mut W_ListObject)).int_items.block = fresh;
        }
        ListStrategy::Float => {
            let fresh = crate::object_array::grow_typed_items_block(
                list.float_items.block,
                4,
                0,
                crate::object_array::gc_float_array_gc_type_id(),
            );
            let obj = crate::gc_roots::shadow_stack_get(root_base);
            (*(obj as *mut W_ListObject)).float_items.block = fresh;
        }
        ListStrategy::Object => {
            let _ = W_ListObject::object_resize_capacity(obj, 4);
        }
        _ => unreachable!("orthodox append fold admits int, float, or object storage"),
    }
    crate::gc_roots::shadow_stack_get(root_base)
}

/// listobject.py IntegerListStrategy.reverse
/// Strategy-preserving: reverses typed storage in place.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_reverse(obj: PyObjectRef) {
    let roots = crate::gc_roots::push_roots();
    let obj = roots.pin_root(obj);
    let list = &mut *(obj as *mut W_ListObject);
    match list.strategy {
        // Empty has nothing to reverse — falls through ListStrategy.reverse
        // (listobject.py defaults) which is a no-op for length 0.
        ListStrategy::Empty | ListStrategy::Size => {}
        ListStrategy::SimpleRange | ListStrategy::Range => {
            let obj = switch_range_to_integer_strategy(list);
            w_list_reverse(obj);
        }
        ListStrategy::Integer => list.int_items.as_mut_slice().reverse(),
        ListStrategy::IntOrFloat => list.int_items.as_mut_slice().reverse(),
        ListStrategy::Float => list.float_items.as_mut_slice().reverse(),
        ListStrategy::Bytes => list.bytes_items.reverse(),
        ListStrategy::Ascii => list.ascii_items.reverse(),
        ListStrategy::Object => list.object_reverse(),
    }
}

/// listobject.py deleteslice (step=1 simple case)
/// Strategy-preserving: drains from typed storage.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_delslice(obj: PyObjectRef, start: usize, end: usize) {
    let roots = crate::gc_roots::push_roots();
    let obj_slot = roots.base();
    let obj = roots.pin_root(obj);
    let list = &mut *(obj as *mut W_ListObject);
    let old_size = list.live_len();
    let mut changed = false;
    match list.strategy {
        // listobject.py EmptyListStrategy.deleteslice is a no-op (pass).
        ListStrategy::Empty | ListStrategy::Size => {}
        ListStrategy::SimpleRange | ListStrategy::Range => {
            let obj = switch_range_to_integer_strategy(list);
            w_list_delslice(obj, start, end);
            return;
        }
        ListStrategy::Integer => {
            let len = list.int_items.len();
            let s = start.min(len);
            let e = end.min(len);
            if s < e {
                list.int_items.drain(s..e);
                changed = true;
            }
        }
        ListStrategy::IntOrFloat => {
            let len = list.int_items.len();
            let s = start.min(len);
            let e = end.min(len);
            if s < e {
                list.int_items.drain(s..e);
                changed = true;
            }
        }
        ListStrategy::Float => {
            let len = list.float_items.len();
            let s = start.min(len);
            let e = end.min(len);
            if s < e {
                list.float_items.drain(s..e);
                changed = true;
            }
        }
        ListStrategy::Bytes => {
            let len = list.bytes_items.len();
            let s = start.min(len);
            let e = end.min(len);
            if s < e {
                list.bytes_items.drain(s..e);
                changed = true;
            }
        }
        ListStrategy::Ascii => {
            let len = list.ascii_items.len();
            let s = start.min(len);
            let e = end.min(len);
            if s < e {
                list.ascii_items.drain(s..e);
                changed = true;
            }
        }
        ListStrategy::Object => {
            let len = list.length_relaxed();
            let s = start.min(len);
            let e = end.min(len);
            if s < e {
                list.object_drain(s..e);
                changed = true;
            }
        }
    }
    if changed {
        // Object storage can cross the before-move safepoint.  The scope's
        // slot, not the pre-barrier `&mut`, names the live list afterwards.
        let list = &mut *(roots.get(obj_slot) as *mut W_ListObject);
        list.sync_allocated(old_size);
    }
}

/// listobject.py IntegerListStrategy._safe_find_or_count
/// Fast path for integer lists: unwrapped comparison.
unsafe fn int_find(items: &[i64], value: i64) -> Option<usize> {
    items.iter().position(|&v| v == value)
}

/// Python int/float cross-type equality: avoids false positives from
/// f64 precision loss (e.g. 2**53+1 != float(2**53)).
#[inline]
fn int_eq_float(ival: i64, fval: f64) -> bool {
    if !fval.is_finite() {
        return false;
    }
    let ival_f = ival as f64;
    if ival_f != fval {
        return false;
    }
    const I64_UPPER_F: f64 = (1u64 << 63) as f64;
    if !(-I64_UPPER_F..I64_UPPER_F).contains(&fval) {
        return false;
    }
    fval as i64 == ival
}

/// Outcome of `W_ListObject.find_or_count` fast path. Mirrors the
/// short-circuit return in `IntegerListStrategy.find_or_count`
/// (listobject.py) and `FloatListStrategy.find_or_count` — when the
/// strategy + needle type match, the typed pool is scanned in place.
/// Otherwise `NeedsGeneric` signals that the caller (pyre-interpreter)
/// must run `ListStrategy.find_or_count`'s generic `space.eq_w` loop.
pub enum ListFindFast {
    /// Fast path applicable, item found at this index (find mode).
    Found(i64),
    /// Fast path applicable, count matched this many times (count mode).
    Count(i64),
    /// Fast path applicable but item not present (find mode).
    NotFound,
    /// Strategy/item type mismatch; caller must run generic eq_w loop.
    NeedsGeneric,
}

/// Typed fast-path for `W_ListObject.find_or_count`. Handles
/// `IntegerListStrategy.find_or_count` (listobject.py) and
/// `FloatListStrategy.find_or_count` (listobject.py) fast paths
/// only. Callers must handle `NeedsGeneric` via the interpreter-level
/// `ListStrategy.find_or_count` which runs the `space.eq_w` loop.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_find_or_count_fast(
    obj: PyObjectRef,
    w_item: PyObjectRef,
    start: i64,
    stop: i64,
    count: bool,
) -> ListFindFast {
    let list = &*(obj as *const W_ListObject);
    match list.strategy {
        // listobject.py EmptyListStrategy.find_or_count: returns
        // `0` in count mode and raises ValueError otherwise. Map the
        // ValueError to NotFound for the find case.
        ListStrategy::Empty | ListStrategy::Size => {
            if count {
                ListFindFast::Count(0)
            } else {
                ListFindFast::NotFound
            }
        }
        ListStrategy::SimpleRange | ListStrategy::Range if is_plain_int1(w_item) => {
            let target = if is_int(w_item) {
                w_int_get_value(w_item)
            } else {
                i64::try_from(w_long_get_value(w_item)).unwrap_or(0)
            };
            let length = range_list_length(list) as i64;
            let (range_start, step) = range_list_start_step(list);
            let delta = (target as i128) - (range_start as i128);
            let step128 = step as i128;
            let candidate = if step128 != 0 && delta % step128 == 0 {
                let index = delta / step128;
                (index >= 0 && index < length as i128).then_some(index as i64)
            } else {
                None
            };
            if let Some(index) = candidate.filter(|&index| start <= index && index < stop) {
                if count {
                    ListFindFast::Count(1)
                } else {
                    ListFindFast::Found(index)
                }
            } else if count {
                ListFindFast::Count(0)
            } else {
                ListFindFast::NotFound
            }
        }
        // listobject.py IntegerListStrategy.find_or_count: fast path
        // when `is_plain_int1(w_obj)`, else fall back to generic.
        ListStrategy::Integer if is_plain_int1(w_item) => {
            let target = if is_int(w_item) {
                w_int_get_value(w_item)
            } else {
                i64::try_from(w_long_get_value(w_item)).unwrap_or(0)
            };
            let items = list.int_items.as_slice();
            let stop = stop.min(items.len() as i64);
            let mut result: i64 = 0;
            let mut i = start.max(0);
            while i < stop {
                if items[i as usize] == target {
                    if count {
                        result += 1;
                    } else {
                        return ListFindFast::Found(i);
                    }
                }
                i += 1;
            }
            if count {
                ListFindFast::Count(result)
            } else {
                ListFindFast::NotFound
            }
        }
        // listobject.py FloatListStrategy.find_or_count → base.
        ListStrategy::Float if is_float_strategy_item(w_item) => {
            let target = w_float_get_value(w_item);
            let items = list.float_items.as_slice();
            let stop = stop.min(items.len() as i64);
            let mut result: i64 = 0;
            let mut i = start.max(0);
            while i < stop {
                let matches = items[i as usize] == target;
                if matches {
                    if count {
                        result += 1;
                    } else {
                        return ListFindFast::Found(i);
                    }
                }
                i += 1;
            }
            if count {
                ListFindFast::Count(result)
            } else {
                ListFindFast::NotFound
            }
        }
        // listobject.py IntOrFloatListStrategy._safe_find_or_count:
        // compare raw longlongs first (same NaN payload), then decoded
        // numeric values (0 == -0.0 and 42 == 42.0).
        ListStrategy::IntOrFloat => {
            let Some(target) = int_or_float_encode_item(w_item) else {
                return ListFindFast::NeedsGeneric;
            };
            let target_float = int_or_float_as_float(target);
            let items = list.int_items.as_slice();
            let stop = stop.min(items.len() as i64);
            let mut result = 0i64;
            let mut i = start.max(0);
            while i < stop {
                let value = items[i as usize];
                if value == target || int_or_float_as_float(value) == target_float {
                    if count {
                        result += 1;
                    } else {
                        return ListFindFast::Found(i);
                    }
                }
                i += 1;
            }
            if count {
                ListFindFast::Count(result)
            } else {
                ListFindFast::NotFound
            }
        }
        _ => ListFindFast::NeedsGeneric,
    }
}

/// listobject.py setslice — strategy-preserving.
///
/// When replacement is a list with the same strategy, operates on typed
/// storage directly. Otherwise falls back to Object strategy.
/// `start` and `end` are already normalized (non-negative, clamped).
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_setslice(
    obj: PyObjectRef,
    start: usize,
    end: usize,
    w_other: PyObjectRef,
) -> Result<(), &'static str> {
    w_list_setslice_mode(obj, start, end, w_other, false)
}

/// `w_list_setslice` with PyPy's `jit.we_are_jitted()` value threaded to the
/// donor's `AbstractUnwrappedStrategy.getitems_copy` fallback.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_list_setslice_mode(
    obj: PyObjectRef,
    start: usize,
    end: usize,
    w_other: PyObjectRef,
    we_are_jitted: bool,
) -> Result<(), &'static str> {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::publish_roots(&[obj, w_other]);
    crate::gc_roots::normalize_roots(root_base, 2);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let w_other = crate::gc_roots::shadow_stack_get(root_base + 1);
    let old_size = (&*(obj as *const W_ListObject)).live_len();
    let result = w_list_setslice_inner(obj, start, end, w_other, we_are_jitted);
    if result.is_ok() {
        let obj = crate::gc_roots::shadow_stack_get(root_base);
        let list = &mut *(obj as *mut W_ListObject);
        if list.live_len() != old_size {
            list.sync_allocated(old_size);
        }
    }
    result
}

unsafe fn w_list_setslice_inner(
    obj: PyObjectRef,
    start: usize,
    end: usize,
    w_other: PyObjectRef,
    we_are_jitted: bool,
) -> Result<(), &'static str> {
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    let _ = crate::gc_roots::pin_root(w_other);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let w_other = crate::gc_roots::shadow_stack_get(root_base + 1);
    let list = &mut *(obj as *mut W_ListObject);
    // A backwards slice (start > stop) is an empty removal, i.e. a pure
    // insertion at `start` — never a negative-length window.
    let end = end.max(start);
    if matches!(
        list.strategy,
        ListStrategy::SimpleRange | ListStrategy::Range
    ) {
        let obj = switch_range_to_integer_strategy(list);
        return w_list_setslice_inner(
            obj,
            start,
            end,
            crate::gc_roots::shadow_stack_get(root_base + 1),
            we_are_jitted,
        );
    }
    if is_list(w_other) {
        let other = &*(w_other as *const W_ListObject);
        // listobject.py EmptyListStrategy.setslice: adopt donor's
        // strategy and storage wholesale. start/end are 0 because list
        // is empty, so this is just "become a copy of w_other".
        if matches!(list.strategy, ListStrategy::Empty | ListStrategy::Size) {
            if matches!(other.strategy, ListStrategy::Empty | ListStrategy::Size) {
                // EmptyListStrategy.copy_into is a no-op. SizeListStrategy
                // inherits it, so the receiver keeps its current strategy
                // object (and shared hint state).
                return Ok(());
            }
            if list.strategy == ListStrategy::Size {
                list.items = std::ptr::null_mut();
            }
            match other.strategy {
                ListStrategy::Empty | ListStrategy::Size => unreachable!("handled above"),
                ListStrategy::SimpleRange | ListStrategy::Range => {
                    let obj = crate::gc_roots::shadow_stack_get(root_base);
                    list_write_barrier(obj);
                    let obj = crate::gc_roots::shadow_stack_get(root_base);
                    let w_other = crate::gc_roots::shadow_stack_get(root_base + 1);
                    let list = &mut *(obj as *mut W_ListObject);
                    let other = &*(w_other as *const W_ListObject);
                    list.strategy = other.strategy;
                    list.items = other.items;
                    return Ok(());
                }
                ListStrategy::Integer => {
                    let fresh = IntArray::from_vec(other.int_items.to_vec());
                    list.int_items.install(fresh);
                    list.strategy = ListStrategy::Integer;
                    return Ok(());
                }
                ListStrategy::IntOrFloat => {
                    let fresh = IntArray::from_vec(other.int_items.to_vec());
                    list.int_items.install(fresh);
                    list.strategy = ListStrategy::IntOrFloat;
                    return Ok(());
                }
                ListStrategy::Float => {
                    let fresh = FloatArray::from_vec(other.float_items.to_vec());
                    list.float_items.install(fresh);
                    list.strategy = ListStrategy::Float;
                    return Ok(());
                }
                ListStrategy::Bytes => {
                    let other =
                        &*(crate::gc_roots::shadow_stack_get(root_base + 1) as *const W_ListObject);
                    let fresh = BytesArray::from_vec(other.bytes_items.to_vec());
                    let obj = W_ListObject::install_bytes_items(
                        crate::gc_roots::shadow_stack_get(root_base),
                        fresh,
                    );
                    let list = &mut *(obj as *mut W_ListObject);
                    list.strategy = ListStrategy::Bytes;
                    return Ok(());
                }
                ListStrategy::Ascii => {
                    let other =
                        &*(crate::gc_roots::shadow_stack_get(root_base + 1) as *const W_ListObject);
                    let fresh = UnicodeArray::from_vec(other.ascii_items.to_vec());
                    let obj = W_ListObject::install_ascii_items(
                        crate::gc_roots::shadow_stack_get(root_base),
                        fresh,
                    );
                    let list = &mut *(obj as *mut W_ListObject);
                    list.strategy = ListStrategy::Ascii;
                    return Ok(());
                }
                ListStrategy::Object => {
                    list.set_object_items_from_vec(other.object_to_vec());
                    let obj = crate::gc_roots::shadow_stack_get(root_base);
                    let list = &mut *(obj as *mut W_ListObject);
                    list.strategy = ListStrategy::Object;
                    list_write_barrier(obj);
                    return Ok(());
                }
            }
        }
        // listobject.py/2013 IntegerListStrategy and :2096/2110
        // FloatListStrategy first generalize themselves when the donor is a
        // compatible numeric strategy, then re-dispatch the same setslice.
        if list.strategy == ListStrategy::Integer && other.strategy == ListStrategy::Range {
            let donated = range_list_values(other);
            let s = start.min(list.int_items.len());
            let e = end.min(list.int_items.len());
            list.int_items.splice(s, e - s, &donated);
            return Ok(());
        }
        if list.strategy == ListStrategy::Integer
            && matches!(
                other.strategy,
                ListStrategy::Float | ListStrategy::IntOrFloat
            )
            && integer_to_int_or_float(list)
        {
            return w_list_setslice_inner(obj, start, end, w_other, we_are_jitted);
        }
        if list.strategy == ListStrategy::Float
            && matches!(
                other.strategy,
                ListStrategy::Integer | ListStrategy::IntOrFloat
            )
            && float_to_int_or_float(list)
        {
            return w_list_setslice_inner(obj, start, end, w_other, we_are_jitted);
        }
        // listobject.py IntOrFloatListStrategy.setslice converts an
        // Integer/Float donor to temporary signed-longlong storage without
        // de-specialising the receiver.
        if list.strategy == ListStrategy::IntOrFloat
            && matches!(other.strategy, ListStrategy::Integer | ListStrategy::Float)
        {
            let converted: Option<Vec<i64>> = match other.strategy {
                ListStrategy::Integer => other
                    .int_items
                    .as_slice()
                    .iter()
                    .map(|&value| int_or_float_encode_int(value))
                    .collect(),
                ListStrategy::Float => other
                    .float_items
                    .as_slice()
                    .iter()
                    .map(|&value| int_or_float_encode_float(value))
                    .collect(),
                _ => unreachable!(),
            };
            if let Some(converted) = converted {
                let s = start.min(list.int_items.len());
                let e = end.min(list.int_items.len());
                list.int_items.splice(s, e - s, &converted);
                return Ok(());
            }
        }
        // listobject.py: not self.list_is_correct_type(w_other) and w_other.length() != 0
        // Only switch strategy when donor is non-empty AND has different type.
        // Empty donor → pure deletion, strategy preserved.
        let other_len = w_list_len(w_other);
        if list.strategy == other.strategy || other_len == 0 {
            match list.strategy {
                ListStrategy::Empty
                | ListStrategy::Size
                | ListStrategy::SimpleRange
                | ListStrategy::Range => unreachable!("handled above"),
                ListStrategy::Integer => {
                    let new_items = if list.strategy == other.strategy {
                        other.int_items.as_slice()
                    } else {
                        &[]
                    };
                    let s = start.min(list.int_items.len());
                    let e = end.min(list.int_items.len());
                    if obj == w_other {
                        let mut v = list.int_items.to_vec();
                        v.splice(s..e, new_items.iter().copied());
                        let fresh = IntArray::from_vec(v);
                        list.int_items.install(fresh);
                    } else {
                        // RPython AbstractUnwrappedStrategy.setslice mutates
                        // the unerased typed storage directly.
                        list.int_items.splice(s, e - s, new_items);
                    }
                    return Ok(());
                }
                ListStrategy::IntOrFloat => {
                    let new_items = if list.strategy == other.strategy {
                        other.int_items.as_slice()
                    } else {
                        &[]
                    };
                    let s = start.min(list.int_items.len());
                    let e = end.min(list.int_items.len());
                    if obj == w_other {
                        let mut v = list.int_items.to_vec();
                        v.splice(s..e, new_items.iter().copied());
                        list.int_items.install(IntArray::from_vec(v));
                    } else {
                        list.int_items.splice(s, e - s, new_items);
                    }
                    return Ok(());
                }
                ListStrategy::Float => {
                    let new_items = if list.strategy == other.strategy {
                        other.float_items.as_slice()
                    } else {
                        &[]
                    };
                    let s = start.min(list.float_items.len());
                    let e = end.min(list.float_items.len());
                    if obj == w_other {
                        let mut v = list.float_items.to_vec();
                        v.splice(s..e, new_items.iter().copied());
                        let fresh = FloatArray::from_vec(v);
                        list.float_items.install(fresh);
                    } else {
                        // RPython AbstractUnwrappedStrategy.setslice mutates
                        // the unerased typed storage directly.
                        list.float_items.splice(s, e - s, new_items);
                    }
                    return Ok(());
                }
                ListStrategy::Bytes => {
                    let obj = crate::gc_roots::shadow_stack_get(root_base);
                    let w_other = crate::gc_roots::shadow_stack_get(root_base + 1);
                    let list = &*(obj as *const W_ListObject);
                    let other = &*(w_other as *const W_ListObject);
                    let donates = list.strategy == other.strategy;
                    let s = start.min(list.bytes_items.len());
                    let e = end.min(list.bytes_items.len());
                    if obj == w_other {
                        let mut values = list.bytes_items.to_vec();
                        let donated = values.clone();
                        values.splice(s..e, donated.into_iter());
                        W_ListObject::install_bytes_items(obj, BytesArray::from_vec(values));
                        return Ok(());
                    }
                    // `splice` may not publish a fresh block itself, so the
                    // room it needs is reserved through the list first.  The
                    // grow collects, so the donor slice is taken after it.
                    let donated = if donates { other.bytes_items.len() } else { 0 };
                    let grown = list.bytes_items.len() - (e - s) + donated;
                    if grown > list.bytes_items.heap_capacity() {
                        W_ListObject::bytes_grow(obj, grown);
                    }
                    let obj = crate::gc_roots::shadow_stack_get(root_base);
                    let w_other = crate::gc_roots::shadow_stack_get(root_base + 1);
                    let list = &mut *(obj as *mut W_ListObject);
                    let other = &*(w_other as *const W_ListObject);
                    let new_items = if donates {
                        other.bytes_items.as_slice()
                    } else {
                        &[]
                    };
                    list.bytes_items.splice(s, e - s, new_items);
                    return Ok(());
                }
                ListStrategy::Ascii => {
                    let obj = crate::gc_roots::shadow_stack_get(root_base);
                    let w_other = crate::gc_roots::shadow_stack_get(root_base + 1);
                    let list = &*(obj as *const W_ListObject);
                    let other = &*(w_other as *const W_ListObject);
                    let donates = list.strategy == other.strategy;
                    let s = start.min(list.ascii_items.len());
                    let e = end.min(list.ascii_items.len());
                    if obj == w_other {
                        let mut values = list.ascii_items.to_vec();
                        let donated = values.clone();
                        values.splice(s..e, donated);
                        W_ListObject::install_ascii_items(obj, UnicodeArray::from_vec(values));
                        return Ok(());
                    }
                    let donated = if donates { other.ascii_items.len() } else { 0 };
                    let grown = list.ascii_items.len() - (e - s) + donated;
                    if grown > list.ascii_items.heap_capacity() {
                        W_ListObject::ascii_grow(obj, grown);
                    }
                    let obj = crate::gc_roots::shadow_stack_get(root_base);
                    let w_other = crate::gc_roots::shadow_stack_get(root_base + 1);
                    let list = &mut *(obj as *mut W_ListObject);
                    let other = &*(w_other as *const W_ListObject);
                    let new_items = if donates {
                        other.ascii_items.as_slice()
                    } else {
                        &[]
                    };
                    list.ascii_items.splice(s, e - s, new_items);
                    return Ok(());
                }
                ListStrategy::Object => {}
            }
        }
    }
    // listobject.py:1751-1753: strategies differ and donor is non-empty →
    // switch to object strategy, then splice as objects.
    let new_items: Vec<PyObjectRef> = if is_list(w_other) {
        let other = &*(w_other as *const W_ListObject);
        temporarily_as_objects(other, we_are_jitted)
    } else {
        return Err("non-list iterable");
    };
    let _new_item_roots = crate::gc_roots::push_roots();
    let new_item_base = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::publish_roots(&new_items);
    crate::gc_roots::normalize_roots(new_item_base, new_items.len());
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let list = &mut *(obj as *mut W_ListObject);
    switch_to_object_strategy(list);
    let mut rooted_new_items = Vec::with_capacity(new_items.len());
    for i in 0..new_items.len() {
        rooted_new_items.push(crate::gc_roots::shadow_stack_get(new_item_base + i));
    }
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    let list = &mut *(obj as *mut W_ListObject);
    let mut v = list.object_to_vec();
    let s = start.min(v.len());
    let e = end.min(v.len());
    v.splice(s..e, rooted_new_items);
    rebuild_object_items(list, v);
    let obj = crate::gc_roots::shadow_stack_get(root_base);
    list_write_barrier(obj);
    Ok(())
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_list_append(list: i64, item: i64) -> i64 {
    unsafe { w_list_append(list as PyObjectRef, item as PyObjectRef) };
    0
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_list_getitem(list: i64, index: i64) -> i64 {
    unsafe {
        match w_list_getitem(list as PyObjectRef, index) {
            Some(value) => value as i64,
            None => panic!("list index out of range in JIT"),
        }
    }
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_list_setitem(list: i64, index: i64, value: i64) -> i64 {
    unsafe {
        if !w_list_setitem(list as PyObjectRef, index, value as PyObjectRef) {
            panic!("list assignment index out of range in JIT");
        }
    }
    0
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_list_reverse(list: i64) -> i64 {
    unsafe { w_list_reverse(list as PyObjectRef) };
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intobject::w_int_new;

    #[test]
    fn strategy_class_names_follow_interp_magic_spellings() {
        assert_eq!(ListStrategy::Empty.class_name(), "EmptyListStrategy");
        assert_eq!(ListStrategy::Size.class_name(), "SizeListStrategy");
        assert_eq!(ListStrategy::Object.class_name(), "ObjectListStrategy");
        assert_eq!(ListStrategy::Integer.class_name(), "IntegerListStrategy");
        assert_eq!(ListStrategy::Float.class_name(), "FloatListStrategy");
        assert_eq!(
            ListStrategy::IntOrFloat.class_name(),
            "IntOrFloatListStrategy"
        );
        assert_eq!(ListStrategy::Bytes.class_name(), "BytesListStrategy");
        assert_eq!(ListStrategy::Ascii.class_name(), "AsciiListStrategy");
        assert_eq!(
            ListStrategy::SimpleRange.class_name(),
            "SimpleRangeListStrategy"
        );
        assert_eq!(ListStrategy::Range.class_name(), "RangeListStrategy");
    }

    #[test]
    fn test_range_list_strategy_creation_and_access() {
        let simple = w_list_new_range(0, 1, 4);
        let range = w_list_new_range(10, -2, 4);
        let empty = w_list_new_range(10, -2, 0);
        unsafe {
            assert_eq!(
                (*(simple as *const W_ListObject)).strategy,
                ListStrategy::SimpleRange
            );
            assert_eq!(
                (*(range as *const W_ListObject)).strategy,
                ListStrategy::Range
            );
            assert_eq!(
                (*(empty as *const W_ListObject)).strategy,
                ListStrategy::Empty
            );
            assert_eq!(w_list_len(simple), 4);
            assert_eq!(w_int_get_value(w_list_getitem(simple, -1).unwrap()), 3);
            assert_eq!(w_int_get_value(w_list_getitem(range, 0).unwrap()), 10);
            assert_eq!(w_int_get_value(w_list_getitem(range, 3).unwrap()), 4);
            assert!(w_list_getitem(range, 4).is_none());
            assert_eq!(w_list_physical_size(simple), None);
            assert!(w_list_resize_hint(simple, 100));
            assert_eq!(w_list_len(simple), 4);
        }
    }

    #[test]
    fn test_range_clone_shares_immutable_state_and_pop_replaces_one_side() {
        let original = w_list_new_range(5, 3, 4);
        unsafe {
            let original_state = (*(original as *const W_ListObject)).items;
            let clone = w_list_clone_if_shared_strategy(original).unwrap();
            assert_eq!((*(clone as *const W_ListObject)).items, original_state);
            assert_eq!(w_int_get_value(w_list_pop(clone, 0).unwrap()), 5);
            assert_eq!(
                (*(clone as *const W_ListObject)).strategy,
                ListStrategy::Range
            );
            assert_ne!((*(clone as *const W_ListObject)).items, original_state);
            assert_eq!(w_list_len(clone), 3);
            assert_eq!(w_int_get_value(w_list_getitem(clone, 0).unwrap()), 8);
            assert_eq!(w_list_len(original), 4);
            assert_eq!(w_int_get_value(w_list_getitem(original, 0).unwrap()), 5);
        }
    }

    #[test]
    fn test_empty_setslice_uses_range_copy_into_storage() {
        let source = w_list_new_range(2, 4, 3);
        let destination = w_list_new(Vec::new());
        unsafe {
            w_list_setslice(destination, 0, 0, source).unwrap();
            assert_eq!(
                (*(destination as *const W_ListObject)).strategy,
                ListStrategy::Range
            );
            assert_eq!(
                (*(destination as *const W_ListObject)).items,
                (*(source as *const W_ListObject)).items
            );
            assert_eq!(w_int_get_value(w_list_getitem(destination, 2).unwrap()), 10);
        }
    }

    #[test]
    fn test_range_mutations_follow_base_range_strategy() {
        let middle = w_list_new_range(10, 2, 5);
        let simple = w_list_new_range(0, 1, 2);
        unsafe {
            assert_eq!(w_int_get_value(w_list_pop(middle, 2).unwrap()), 14);
            assert_eq!(
                (*(middle as *const W_ListObject)).strategy,
                ListStrategy::Integer
            );
            assert_eq!(w_int_get_value(w_list_pop_end(simple).unwrap()), 1);
            assert_eq!(
                (*(simple as *const W_ListObject)).strategy,
                ListStrategy::SimpleRange
            );
            assert_eq!(w_int_get_value(w_list_pop_end(simple).unwrap()), 0);
            assert_eq!(
                (*(simple as *const W_ListObject)).strategy,
                ListStrategy::Empty
            );

            let append_int = w_list_new_range(0, 1, 3);
            w_list_append(append_int, w_int_new(3));
            assert_eq!(
                (*(append_int as *const W_ListObject)).strategy,
                ListStrategy::Integer
            );
            let append_object = w_list_new_range(0, 1, 3);
            w_list_append(append_object, crate::noneobject::w_none());
            assert_eq!(
                (*(append_object as *const W_ListObject)).strategy,
                ListStrategy::Object
            );
        }
    }

    #[test]
    fn test_range_sort_preserves_only_an_already_ordered_range() {
        let ascending = w_list_new_range(0, 1, 4);
        let descending = w_list_new_range(9, -2, 4);
        unsafe {
            assert!(w_list_sort_range(ascending, false));
            assert_eq!(
                (*(ascending as *const W_ListObject)).strategy,
                ListStrategy::SimpleRange
            );
            assert!(!w_list_sort_range(ascending, true));
            assert_eq!(
                (*(ascending as *const W_ListObject)).strategy,
                ListStrategy::Integer
            );

            assert!(w_list_sort_range(descending, true));
            assert_eq!(
                (*(descending as *const W_ListObject)).strategy,
                ListStrategy::Range
            );
            assert!(!w_list_sort_range(descending, false));
            assert_eq!(
                (*(descending as *const W_ListObject)).strategy,
                ListStrategy::Integer
            );
        }
    }

    #[test]
    fn test_list_create_and_access() {
        let items = vec![w_int_new(10), w_int_new(20), w_int_new(30)];
        let list = w_list_new(items);
        unsafe {
            assert!(is_list(list));
            assert_eq!(w_list_len(list), 3);
            let item = w_list_getitem(list, 0).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 10);
            let item = w_list_getitem(list, 2).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 30);
        }
    }

    #[test]
    fn is_plain_int1_rejects_a_bigint_that_does_not_fit_i64() {
        // listobject.py is_plain_int1 / IntegerListStrategy.is_correct_type:
        // a W_LongObject is only a plain int when `_fits_int()` holds.
        // 2**70 must stay on the object-strategy / Generic sort path.
        let big = crate::longobject::w_long_new(
            majit_rlib::rbigint::RBigInt::fromint(i64::MAX)
                + majit_rlib::rbigint::RBigInt::fromint(1),
        );
        let bigger = crate::longobject::w_long_new(majit_rlib::rbigint::RBigInt::fromint(1) << 70);
        unsafe {
            assert!(crate::pyobject::is_long(big));
            assert!(!is_plain_int1(big));
            assert!(crate::pyobject::is_long(bigger));
            assert!(!is_plain_int1(bigger));
            assert!(!w_list_uses_int_storage(w_list_new(vec![
                bigger,
                w_int_new(5)
            ])));
        }
    }

    #[test]
    fn unused_typed_storage_holds_no_block() {
        // An Object-strategy list reads neither typed array, and an
        // Integer-strategy list reads only `int_items`: the other side must
        // carry the empty form, not an allocated single-slot block. This is the
        // shape `emit_typed_list_inline` already leaves for traced code.
        let object_list = w_list_new(vec![crate::w_none()]);
        let int_list = w_list_new(vec![w_int_new(1)]);
        let float_list = w_list_new(vec![crate::floatobject::w_float_new(1.5)]);
        unsafe {
            let l = &*(object_list as *const W_ListObject);
            assert_eq!(l.strategy, ListStrategy::Object);
            assert!(l.int_items.block.is_null());
            assert!(l.float_items.block.is_null());
            assert!(l.bytes_items.block.is_null());
            assert!(l.ascii_items.block.is_null());
            assert!(l.int_items.as_slice().is_empty());
            assert!(l.float_items.as_slice().is_empty());

            let l = &*(int_list as *const W_ListObject);
            assert_eq!(l.strategy, ListStrategy::Integer);
            assert!(!l.int_items.block.is_null());
            assert!(l.float_items.block.is_null());
            assert!(l.bytes_items.block.is_null());
            assert!(l.ascii_items.block.is_null());

            let l = &*(float_list as *const W_ListObject);
            assert_eq!(l.strategy, ListStrategy::Float);
            assert!(l.int_items.block.is_null());
            assert!(!l.float_items.block.is_null());
            assert!(l.bytes_items.block.is_null());
            assert!(l.ascii_items.block.is_null());
        }
    }

    #[test]
    fn bytes_strategy_stores_erased_blocks_and_dehomogenizes() {
        let a = crate::bytesobject::w_bytes_from_bytes(b"a");
        let b = crate::bytesobject::w_bytes_from_bytes(b"b");
        let list = w_list_new(vec![a, b]);
        unsafe {
            let l = &*(list as *const W_ListObject);
            assert_eq!(l.strategy, ListStrategy::Bytes);
            assert!(l.items.is_null());
            assert!(!l.bytes_items.block.is_null());
            assert_eq!(
                crate::bytesobject::w_bytes_data(w_list_getitem(list, 0).unwrap()),
                b"a"
            );

            let c = crate::bytesobject::w_bytes_from_bytes(b"c");
            w_list_append(list, c);
            assert_eq!(
                (*(list as *const W_ListObject)).strategy,
                ListStrategy::Bytes
            );
            assert_eq!(w_list_len(list), 3);

            w_list_append(list, crate::w_str_new("not bytes"));
            let l = &*(list as *const W_ListObject);
            assert_eq!(l.strategy, ListStrategy::Object);
            assert!(l.bytes_items.block.is_null());
            assert_eq!(w_list_len(list), 4);
        }
    }

    #[test]
    fn ascii_strategy_stores_utf8_payloads_and_dehomogenizes() {
        let a = crate::w_str_new("alpha");
        let b = crate::w_str_new("beta");
        let a_storage = unsafe { w_str_storage(a) };
        let list = w_list_new(vec![a, b]);
        unsafe {
            let l = &*(list as *const W_ListObject);
            assert_eq!(l.strategy, ListStrategy::Ascii);
            assert!(l.items.is_null());
            assert!(!l.ascii_items.block.is_null());
            let wrapped = w_list_getitem(list, 0).unwrap();
            assert_eq!(crate::w_str_get_value(wrapped), "alpha");
            assert_eq!(w_str_storage(wrapped), a_storage);

            w_list_append(list, crate::w_str_new("gamma"));
            assert_eq!(
                (*(list as *const W_ListObject)).strategy,
                ListStrategy::Ascii
            );
            w_list_append(list, crate::w_str_new("é"));
            let l = &*(list as *const W_ListObject);
            assert_eq!(l.strategy, ListStrategy::Object);
            assert!(l.ascii_items.block.is_null());
            assert_eq!(w_list_len(list), 4);
            assert_eq!(w_str_storage(w_list_getitem(list, 0).unwrap()), a_storage);
        }
    }

    #[test]
    fn unboxed_getitems_copy_reuses_consecutive_equal_wrappers() {
        // listobject.py AbstractUnwrappedStrategy.getitems_copy: each typed
        // strategy applies `_quick_cmp` and retains the preceding wrapper.
        let integer = w_int_new(1_000_000);
        let float = crate::floatobject::w_float_new(1.25);
        let bytes = crate::bytesobject::w_bytes_from_bytes(b"shared");
        let ascii = crate::w_str_new("shared");
        for (name, list) in [
            ("IntegerListStrategy", w_list_new(vec![integer, integer])),
            ("FloatListStrategy", w_list_new(vec![float, float])),
            ("BytesListStrategy", w_list_new(vec![bytes, bytes])),
            ("AsciiListStrategy", w_list_new(vec![ascii, ascii])),
        ] {
            let copied = unsafe { w_list_items_copy_as_vec(list) };
            assert_eq!(copied.len(), 2);
            assert!(std::ptr::eq(copied[0], copied[1]), "{name}");

            let traced = unsafe { w_list_items_copy_as_vec_mode(list, true) };
            assert_eq!(traced.len(), 2);
            assert!(
                !std::ptr::eq(traced[0], traced[1]),
                "{name} under we_are_jitted"
            );
        }
    }

    #[test]
    fn ascii_strategy_mutations_and_sort_stay_unwrapped() {
        let list = w_list_new(vec![crate::w_str_new("c"), crate::w_str_new("a")]);
        let donor = w_list_new(vec![crate::w_str_new("b")]);
        unsafe {
            w_list_insert(list, 1, crate::w_str_new("d"));
            assert_eq!(
                (*(list as *const W_ListObject)).strategy,
                ListStrategy::Ascii
            );
            w_list_delslice(list, 1, 2);
            w_list_setslice(list, 1, 1, donor).unwrap();
            assert!(w_list_sort_strings(list, false));
            assert_eq!(
                crate::w_str_get_value(w_list_getitem(list, 0).unwrap()),
                "a"
            );
            assert_eq!(
                crate::w_str_get_value(w_list_getitem(list, 1).unwrap()),
                "b"
            );
            assert_eq!(
                crate::w_str_get_value(w_list_getitem(list, 2).unwrap()),
                "c"
            );
            w_list_reverse(list);
            assert_eq!(crate::w_str_get_value(w_list_pop(list, 0).unwrap()), "c");
            assert_eq!(
                (*(list as *const W_ListObject)).strategy,
                ListStrategy::Ascii
            );

            let empty = w_list_new(Vec::new());
            w_list_setslice(empty, 0, 0, list).unwrap();
            assert_eq!(
                (*(empty as *const W_ListObject)).strategy,
                ListStrategy::Ascii
            );
            assert_eq!(w_list_len(empty), 2);
        }
    }

    #[test]
    fn empty_typed_storage_grows_on_first_append() {
        // The empty form has capacity 0, so the first write must reach `grow`
        // rather than the no-resize leg.
        let mut arr = IntArray::empty();
        assert_eq!(arr.len(), 0);
        assert_eq!(arr.heap_capacity(), 0);
        assert_eq!(arr.spare_capacity(), 0);
        arr.push(7);
        assert!(!arr.block.is_null());
        assert_eq!(arr.as_slice(), &[7]);

        let mut arr = FloatArray::empty();
        assert_eq!(arr.heap_capacity(), 0);
        arr.push(2.5);
        assert_eq!(arr.as_slice(), &[2.5]);
    }

    #[test]
    fn clear_drops_typed_storage_to_the_empty_form() {
        let list = w_list_new(vec![w_int_new(1), w_int_new(2)]);
        unsafe {
            w_list_clear(list);
            let l = &*(list as *const W_ListObject);
            assert_eq!(l.strategy, ListStrategy::Empty);
            assert!(l.int_items.block.is_null());
            assert!(l.float_items.block.is_null());
            assert!(l.bytes_items.block.is_null());
            assert!(l.ascii_items.block.is_null());
            // The next append reinstalls the matching typed storage.
            w_list_append(list, w_int_new(9));
            let l = &*(list as *const W_ListObject);
            assert_eq!(l.strategy, ListStrategy::Integer);
            assert_eq!(w_list_len(list), 1);
        }
    }

    #[test]
    fn test_list_negative_index() {
        let items = vec![w_int_new(1), w_int_new(2), w_int_new(3)];
        let list = w_list_new(items);
        unsafe {
            let item = w_list_getitem(list, -1).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 3);
        }
    }

    #[test]
    fn integer_strategy_oopspec_leaves_roundtrip() {
        let items = vec![w_int_new(10), w_int_new(20), w_int_new(30)];
        let list = w_list_new(items);
        unsafe {
            let l = &mut *(list as *mut W_ListObject);
            assert_eq!(l.strategy, ListStrategy::Integer);
            assert_eq!(ll_list_int_length(l), 3);
            assert_eq!(ll_list_int_getitem_fast(l, 0), 10);
            assert_eq!(ll_list_int_getitem_fast(l, 2), 30);
            ll_list_int_setitem_fast(l, 1, 99);
            assert_eq!(ll_list_int_getitem_fast(l, 1), 99);
            // The write is observable through the public accessor.
            let item = w_list_getitem(list, 1).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 99);
        }
    }

    #[test]
    fn integer_strategy_oopspec_tags_present() {
        // The `#[oopspec(...)]` attribute emits the spec string for the
        // codewriter's `_handle_list_call` to decode (rlib/jit.py:250 parity).
        assert_eq!(oopspec_ll_list_int_length, "list.int_len(l)");
        assert_eq!(
            oopspec_ll_list_int_getitem_fast,
            "list.int_getitem(l, index)"
        );
        assert_eq!(
            oopspec_ll_list_int_setitem_fast,
            "list.int_setitem(l, index, item)"
        );
        assert_eq!(oopspec_ll_list_int_capacity, "list.int_capacity(l)");
        assert_eq!(oopspec_ll_list_int_set_len, "list.int_set_len(l, n)");
    }

    #[test]
    fn object_strategy_oopspec_tags_present() {
        assert_eq!(oopspec_ll_list_obj_length, "list.obj_len(l)");
        assert_eq!(oopspec_ll_list_obj_capacity, "list.obj_capacity(l)");
        assert_eq!(oopspec_ll_list_obj_set_len, "list.obj_set_len(l, n)");
        assert_eq!(
            oopspec_ll_list_obj_setitem_fast,
            "list.obj_setitem(l, index, item)"
        );
    }

    #[test]
    fn test_list_setitem() {
        let items = vec![w_int_new(1), w_int_new(2)];
        let list = w_list_new(items);
        unsafe {
            assert!(w_list_setitem(list, 0, w_int_new(99)));
            let item = w_list_getitem(list, 0).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 99);
        }
    }

    #[test]
    fn test_list_append() {
        let list = w_list_new(vec![]);
        unsafe {
            w_list_append(list, w_int_new(42));
            assert_eq!(w_list_len(list), 1);
            let item = w_list_getitem(list, 0).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(item), 42);
        }
    }

    #[test]
    fn test_list_out_of_bounds() {
        let list = w_list_new(vec![w_int_new(1)]);
        unsafe {
            assert!(w_list_getitem(list, 5).is_none());
            assert!(w_list_getitem(list, -5).is_none());
            assert!(!w_list_setitem(list, 5, w_int_new(0)));
        }
    }

    #[test]
    fn test_jit_list_helpers_share_list_semantics() {
        let list = w_list_new(vec![w_int_new(1), w_int_new(2)]);
        unsafe {
            assert_eq!(
                crate::intobject::w_int_get_value(jit_list_getitem(list as i64, 1) as PyObjectRef),
                2
            );
        }
        assert_eq!(jit_list_setitem(list as i64, 0, w_int_new(9) as i64), 0);
        assert_eq!(jit_list_append(list as i64, w_int_new(7) as i64), 0);
        unsafe {
            assert_eq!(w_list_len(list), 3);
            assert_eq!(
                crate::intobject::w_int_get_value(w_list_getitem(list, 0).unwrap()),
                9
            );
            assert_eq!(
                crate::intobject::w_int_get_value(w_list_getitem(list, 2).unwrap()),
                7
            );
        }
    }

    #[test]
    fn test_w_list_pop_normalizes_negative_index() {
        let list = w_list_new(vec![w_int_new(10), w_int_new(20), w_int_new(30)]);
        unsafe {
            let popped = w_list_pop(list, -1).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(popped), 30);
            assert_eq!(w_list_len(list), 2);
        }
    }

    #[test]
    fn test_w_list_pop_out_of_range_returns_none() {
        // An out-of-range index leaves the list untouched and returns
        // `None` (the caller raises IndexError).
        let list = w_list_new(vec![w_int_new(10)]);
        unsafe {
            assert!(w_list_pop(list, 5).is_none());
            assert!(w_list_pop(list, -5).is_none());
            assert_eq!(w_list_len(list), 1);
        }
    }

    #[test]
    fn test_w_list_pop_end_returns_none_for_empty_every_strategy() {
        let empty = w_list_new(Vec::new());
        let integer = w_list_new(vec![w_int_new(1)]);
        let float = w_list_new(vec![crate::floatobject::w_float_new(1.0)]);
        let object = w_list_new_object(vec![w_int_new(1)]);

        unsafe {
            assert!(w_list_pop_end(integer).is_some());
            assert!(w_list_pop_end(float).is_some());
            assert!(w_list_pop_end(object).is_some());

            assert!(w_list_uses_empty_storage(empty));
            assert!(w_list_uses_int_storage(integer));
            assert!(w_list_uses_float_storage(float));
            assert!(w_list_uses_object_storage(object));

            assert!(w_list_pop_end(empty).is_none());
            assert!(w_list_pop_end(integer).is_none());
            assert!(w_list_pop_end(float).is_none());
            assert!(w_list_pop_end(object).is_none());
        }
    }

    #[test]
    fn test_list_uses_integer_strategy_for_homogeneous_ints() {
        let list = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            assert!(!w_list_uses_object_storage(list));
            assert_eq!(w_list_len(list), 3);
        }
    }

    #[test]
    fn test_list_setitem_mixed_value_switches_to_int_or_float_strategy() {
        let list = w_list_new(vec![w_int_new(1), w_int_new(2)]);
        let float = crate::floatobject::w_float_new(3.5);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            assert!(w_list_setitem(list, 0, float));
            assert!(w_list_uses_int_or_float_storage(list));
            let value = w_list_getitem(list, 0).unwrap();
            assert!(crate::pyobject::is_float(value));
        }
    }

    #[test]
    fn test_list_append_mixed_value_switches_to_int_or_float_strategy() {
        let list = w_list_new(vec![w_int_new(1), w_int_new(2)]);
        let float = crate::floatobject::w_float_new(3.5);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            w_list_append(list, float);
            assert!(w_list_uses_int_or_float_storage(list));
            assert_eq!(w_list_len(list), 3);
            let value = w_list_getitem(list, 2).unwrap();
            assert!(crate::pyobject::is_float(value));
        }
    }

    #[test]
    fn test_int_or_float_strategy_preserves_types_and_numeric_equality() {
        let list = w_list_new(vec![
            w_int_new(42),
            crate::floatobject::w_float_new(42.0),
            crate::floatobject::w_float_new(-0.0),
        ]);
        unsafe {
            assert!(w_list_uses_int_or_float_storage(list));
            let integer = w_list_getitem(list, 0).unwrap();
            let float = w_list_getitem(list, 1).unwrap();
            let negative_zero = w_list_getitem(list, 2).unwrap();
            assert!(crate::pyobject::is_int(integer));
            assert!(crate::pyobject::is_float(float));
            assert_eq!(crate::intobject::w_int_get_value(integer), 42);
            assert_eq!(crate::floatobject::w_float_get_value(float), 42.0);
            assert!(crate::floatobject::w_float_get_value(negative_zero).is_sign_negative());

            assert!(matches!(
                w_list_find_or_count_fast(list, w_int_new(42), 0, 3, true),
                ListFindFast::Count(2)
            ));
            assert!(matches!(
                w_list_find_or_count_fast(list, crate::floatobject::w_float_new(0.0), 0, 3, false),
                ListFindFast::Found(2)
            ));
        }
    }

    #[test]
    fn test_int_or_float_rejects_out_of_int32_range() {
        let list = w_list_new(vec![w_int_new(i32::MAX as i64 + 1), w_int_new(1)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            w_list_append(list, crate::floatobject::w_float_new(2.5));
            assert!(w_list_uses_object_storage(list));
        }
    }

    #[test]
    fn test_int_or_float_setslice_accepts_integer_and_float_strategies() {
        let list = w_list_new(vec![w_int_new(1), crate::floatobject::w_float_new(4.0)]);
        let integers = w_list_new(vec![w_int_new(2), w_int_new(3)]);
        let floats = w_list_new(vec![crate::floatobject::w_float_new(2.5)]);
        unsafe {
            w_list_setslice(list, 1, 1, integers).unwrap();
            assert!(w_list_uses_int_or_float_storage(list));
            w_list_setslice(list, 1, 3, floats).unwrap();
            assert!(w_list_uses_int_or_float_storage(list));
            assert_eq!(w_list_len(list), 3);
            assert_eq!(
                crate::floatobject::w_float_get_value(w_list_getitem(list, 1).unwrap()),
                2.5
            );
            assert_eq!(
                crate::floatobject::w_float_get_value(w_list_getitem(list, 2).unwrap()),
                4.0
            );
        }
    }

    #[test]
    fn test_list_uses_float_strategy_for_homogeneous_floats() {
        let list = w_list_new(vec![
            crate::floatobject::w_float_new(1.25),
            crate::floatobject::w_float_new(2.5),
            crate::floatobject::w_float_new(3.75),
        ]);
        unsafe {
            assert!(w_list_uses_float_storage(list));
            assert!(!w_list_uses_object_storage(list));
            assert_eq!(w_list_len(list), 3);
            let value = w_list_getitem(list, 1).unwrap();
            assert!(crate::pyobject::is_float(value));
            assert_eq!(crate::floatobject::w_float_get_value(value), 2.5);
        }
    }

    #[test]
    fn test_list_setitem_mixed_on_float_strategy_switches_to_int_or_float_strategy() {
        let list = w_list_new(vec![
            crate::floatobject::w_float_new(1.0),
            crate::floatobject::w_float_new(2.0),
        ]);
        unsafe {
            assert!(w_list_uses_float_storage(list));
            assert!(w_list_setitem(list, 0, w_int_new(7)));
            assert!(w_list_uses_int_or_float_storage(list));
            let value = w_list_getitem(list, 0).unwrap();
            assert!(crate::pyobject::is_int(value));
        }
    }

    #[test]
    fn test_list_append_mixed_on_float_strategy_switches_to_int_or_float_strategy() {
        let list = w_list_new(vec![
            crate::floatobject::w_float_new(1.0),
            crate::floatobject::w_float_new(2.0),
        ]);
        unsafe {
            assert!(w_list_uses_float_storage(list));
            w_list_append(list, w_int_new(7));
            assert!(w_list_uses_int_or_float_storage(list));
            assert_eq!(w_list_len(list), 3);
            let value = w_list_getitem(list, 2).unwrap();
            assert!(crate::pyobject::is_int(value));
        }
    }

    // ── per-strategy operation tests ─────────────────────────────────────────
    // These verify that pop/pop_end/insert/reverse/clear/delslice do NOT
    // switch to ObjectStrategy when the list is homogeneous (int or float).

    #[test]
    fn test_int_list_pop_stays_integer_strategy() {
        // AbstractUnwrappedStrategy.pop (listobject.py)
        let list = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            let popped = w_list_pop(list, 1).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(popped), 2);
            assert!(
                w_list_uses_int_storage(list),
                "pop must not switch strategy"
            );
            assert_eq!(w_list_len(list), 2);
            assert_eq!(
                crate::intobject::w_int_get_value(w_list_getitem(list, 0).unwrap()),
                1
            );
            assert_eq!(
                crate::intobject::w_int_get_value(w_list_getitem(list, 1).unwrap()),
                3
            );
        }
    }

    #[test]
    fn test_int_list_pop_end_stays_integer_strategy() {
        // AbstractUnwrappedStrategy.pop_end (listobject.py)
        let list = w_list_new(vec![w_int_new(10), w_int_new(20)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            let popped = w_list_pop_end(list).unwrap();
            assert_eq!(crate::intobject::w_int_get_value(popped), 20);
            assert!(
                w_list_uses_int_storage(list),
                "pop_end must not switch strategy"
            );
            assert_eq!(w_list_len(list), 1);
        }
    }

    #[test]
    fn test_int_list_insert_stays_integer_strategy() {
        // AbstractUnwrappedStrategy.insert (listobject.py)
        let list = w_list_new(vec![w_int_new(1), w_int_new(3)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            w_list_insert(list, 1, w_int_new(2));
            assert!(
                w_list_uses_int_storage(list),
                "insert int must not switch strategy"
            );
            assert_eq!(w_list_len(list), 3);
            assert_eq!(
                crate::intobject::w_int_get_value(w_list_getitem(list, 1).unwrap()),
                2
            );
        }
    }

    #[test]
    fn test_int_list_insert_float_switches_to_int_or_float() {
        // AbstractUnwrappedStrategy.switch_to_next_strategy (listobject.py)
        let list = w_list_new(vec![w_int_new(1), w_int_new(2)]);
        let fv = crate::floatobject::w_float_new(9.0);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            w_list_insert(list, 1, fv);
            assert!(w_list_uses_int_or_float_storage(list));
            assert_eq!(w_list_len(list), 3);
        }
    }

    #[test]
    fn test_int_list_reverse_stays_integer_strategy() {
        // AbstractUnwrappedStrategy.reverse (listobject.py)
        let list = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            w_list_reverse(list);
            assert!(
                w_list_uses_int_storage(list),
                "reverse must not switch strategy"
            );
            assert_eq!(
                crate::intobject::w_int_get_value(w_list_getitem(list, 0).unwrap()),
                3
            );
            assert_eq!(
                crate::intobject::w_int_get_value(w_list_getitem(list, 2).unwrap()),
                1
            );
        }
    }

    #[test]
    fn test_new_empty_uses_empty_strategy() {
        // listobject.py fresh empty list uses EmptyListStrategy.
        let list = w_list_new(Vec::new());
        unsafe {
            assert!(w_list_uses_empty_storage(list));
            assert_eq!(w_list_len(list), 0);
        }
    }

    #[test]
    fn test_size_list_strategy_consumes_hint_on_first_append() {
        // listobject.py SizeListStrategy inherits EmptyListStrategy and only
        // overrides get_sizehint/_resize_hint.
        let list = w_list_new_with_sizehint(13);
        unsafe {
            assert_eq!(
                (*(list as *const W_ListObject)).strategy,
                ListStrategy::Size
            );
            assert_eq!(w_list_len(list), 0);
            assert_eq!(w_list_physical_size(list), Some(0));
            w_list_append(list, w_int_new(7));
            assert_eq!(
                (*(list as *const W_ListObject)).strategy,
                ListStrategy::Integer
            );
            assert_eq!(w_list_physical_size(list), Some(13));
            assert_eq!(w_list_len(list), 1);
            assert_eq!(w_int_get_value(w_list_getitem(list, 0).unwrap()), 7);
        }
    }

    #[test]
    fn test_size_list_strategy_clone_shares_strategy_state() {
        // SizeListStrategy inherits EmptyListStrategy.clone, which retains
        // `self` rather than constructing a new strategy object.
        unsafe {
            let list = w_list_new_with_sizehint(5);
            let clone = w_list_clone_if_shared_strategy(list).unwrap();
            assert!(w_list_resize_hint(list, 9));
            w_list_append(clone, w_int_new(7));
            assert_eq!(w_list_physical_size(clone), Some(9));
        }
    }

    #[test]
    fn test_object_list_sizehint_preallocates_without_changing_length() {
        // BaseObjSpace._unpackiterable_unknown_length uses RPython
        // newlist_hint for a raw list[W_Root], not SizeListStrategy.
        unsafe {
            let list = w_list_new_object_with_sizehint(11);
            assert_eq!(
                (*(list as *const W_ListObject)).strategy,
                ListStrategy::Object
            );
            assert_eq!(w_list_len(list), 0);
            assert_eq!(w_list_physical_size(list), Some(11));
            w_list_append(list, w_int_new(7));
            assert_eq!(w_list_physical_size(list), Some(11));

            let empty = w_list_new_object_with_sizehint(0);
            assert_eq!(w_list_physical_size(empty), Some(0));
        }
    }

    #[test]
    fn test_size_list_strategy_preallocates_each_unwrapped_storage() {
        unsafe {
            let float = w_list_new_with_sizehint(5);
            w_list_append(float, w_float_new(1.5));
            assert_eq!(
                (*(float as *const W_ListObject)).strategy,
                ListStrategy::Float
            );
            assert_eq!(w_list_physical_size(float), Some(5));

            let bytes = w_list_new_with_sizehint(6);
            w_list_append(bytes, crate::bytesobject::w_bytes_from_bytes(b"x"));
            assert_eq!(
                (*(bytes as *const W_ListObject)).strategy,
                ListStrategy::Bytes
            );
            assert_eq!(w_list_physical_size(bytes), Some(6));

            let ascii = w_list_new_with_sizehint(7);
            w_list_append(ascii, crate::unicodeobject::w_str_new("x"));
            assert_eq!(
                (*(ascii as *const W_ListObject)).strategy,
                ListStrategy::Ascii
            );
            assert_eq!(w_list_physical_size(ascii), Some(7));

            let object = w_list_new_with_sizehint(8);
            w_list_append(object, crate::w_none());
            assert_eq!(
                (*(object as *const W_ListObject)).strategy,
                ListStrategy::Object
            );
            assert_eq!(w_list_physical_size(object), Some(8));
        }
    }

    #[test]
    fn test_resize_hint_uses_rpython_capacity_policy() {
        unsafe {
            let empty = w_list_new(Vec::new());
            assert!(w_list_resize_hint(empty, 13));
            assert_eq!(
                (*(empty as *const W_ListObject)).strategy,
                ListStrategy::Size
            );
            assert_eq!(w_list_sizehint(empty), Some(13));

            let ints = w_list_new(vec![w_int_new(1), w_int_new(2)]);
            assert_eq!(w_list_physical_size(ints), Some(2));
            assert!(w_list_resize_hint(ints, 10));
            // rlist.py: newsize + 6 + (newsize >> 3)
            assert_eq!(w_list_physical_size(ints), Some(17));
            assert_eq!(w_int_get_value(w_list_getitem(ints, 1).unwrap()), 2);
        }
    }

    #[test]
    fn test_clear_resets_to_empty_strategy() {
        // listobject.py W_ListObject.clear → EmptyListStrategy.
        let list = w_list_new(vec![w_int_new(1), w_int_new(2)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            w_list_clear(list);
            assert!(
                w_list_uses_empty_storage(list),
                "clear must switch to EmptyListStrategy"
            );
            assert_eq!(w_list_len(list), 0);
        }
    }

    #[test]
    fn test_empty_first_int_append_switches_to_int_strategy() {
        // listobject.py EmptyListStrategy.append picks the typed strategy
        // matching the first item.
        let list = w_list_new(Vec::new());
        unsafe {
            assert!(w_list_uses_empty_storage(list));
            w_list_append(list, w_int_new(7));
            assert!(w_list_uses_int_storage(list));
            assert_eq!(w_list_len(list), 1);
            assert_eq!(
                crate::intobject::w_int_get_value(w_list_getitem(list, 0).unwrap()),
                7
            );
        }
    }

    #[test]
    fn test_empty_first_float_append_switches_to_float_strategy() {
        let list = w_list_new(Vec::new());
        unsafe {
            assert!(w_list_uses_empty_storage(list));
            w_list_append(list, crate::floatobject::w_float_new(2.5));
            assert!(w_list_uses_float_storage(list));
            assert_eq!(w_list_len(list), 1);
        }
    }

    #[test]
    fn test_int_list_delslice_stays_integer_strategy() {
        // AbstractUnwrappedStrategy.deleteslice (listobject.py)
        let list = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3), w_int_new(4)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
            w_list_delslice(list, 1, 3);
            assert!(
                w_list_uses_int_storage(list),
                "delslice must not switch strategy"
            );
            assert_eq!(w_list_len(list), 2);
            assert_eq!(
                crate::intobject::w_int_get_value(w_list_getitem(list, 0).unwrap()),
                1
            );
            assert_eq!(
                crate::intobject::w_int_get_value(w_list_getitem(list, 1).unwrap()),
                4
            );
        }
    }

    #[test]
    fn test_float_list_pop_stays_float_strategy() {
        // AbstractUnwrappedStrategy.pop (listobject.py)
        let list = w_list_new(vec![
            crate::floatobject::w_float_new(1.0),
            crate::floatobject::w_float_new(2.0),
            crate::floatobject::w_float_new(3.0),
        ]);
        unsafe {
            assert!(w_list_uses_float_storage(list));
            let popped = w_list_pop(list, 0).unwrap();
            assert_eq!(crate::floatobject::w_float_get_value(popped), 1.0);
            assert!(
                w_list_uses_float_storage(list),
                "pop must not switch strategy"
            );
            assert_eq!(w_list_len(list), 2);
        }
    }

    #[test]
    fn test_float_list_reverse_stays_float_strategy() {
        // AbstractUnwrappedStrategy.reverse (listobject.py)
        let list = w_list_new(vec![
            crate::floatobject::w_float_new(1.0),
            crate::floatobject::w_float_new(2.0),
        ]);
        unsafe {
            assert!(w_list_uses_float_storage(list));
            w_list_reverse(list);
            assert!(
                w_list_uses_float_storage(list),
                "reverse must not switch strategy"
            );
            assert_eq!(
                crate::floatobject::w_float_get_value(w_list_getitem(list, 0).unwrap()),
                2.0
            );
        }
    }
}
