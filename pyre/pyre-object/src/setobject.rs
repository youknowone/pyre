//! W_SetObject — Python `set` type.
//!
//! PyPy equivalent: pypy/objspace/std/setobject.py
//!
//! Stores arbitrary PyObjectRef elements in a hashed [`rordereddict`] table of
//! ObjectKey,
//! reusing the dict object strategy's hashing and equality semantics.
//! `setobject.py SetStrategy` is the dispatch. A fresh set is
//! `EmptySetStrategy` (`sstorage` is `erase(None)`). The first `add` installs
//! `IntegerSetStrategy` for a plain int (`is_plain_int1`) and
//! `ObjectSetStrategy` otherwise. Bytes, ascii, and identity are later steps.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;
use pyre_macros::pyre_class;
use std::cell::UnsafeCell;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicUsize, Ordering};

pub static SET_TYPE: PyType = crate::pyobject::new_pytype("set");
pub static FROZENSET_TYPE: PyType = crate::pyobject::new_pytype("frozenset");

/// setobject.py `W_SetIterObject`.  Unlike the old sequence-iterator
/// adapter this keeps the live set, so a size change is observed by next().
#[pyre_class("set_iterator", static_name = "SET_ITERATOR")]
pub struct W_SetIterObject {
    pub w_set: PyObjectRef,
    pub startlen: usize,
    /// Elements handed out so far, which `descr_reduce` needs and the slot
    /// cursor no longer counts once a hole opens.
    pub index: usize,
    /// Where the next element is read from.  `index` counted table positions
    /// while a delete renumbered them; a tombstoned table renumbers nothing,
    /// so the cursor is a slot and the two part company after a `discard`.
    pub slot: usize,
}

#[expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "PyObjectRef is a GC-managed VM handle whose validity is established at the interpreter boundary; this item is the safe object-space facade"
)]
pub fn w_set_iter_new(w_set: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let w_set = crate::gc_roots::pin_root(w_set);
    let startlen = unsafe { w_set_len(w_set) };
    W_SetIterObject::allocate_stable(W_SetIterObject {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_set,
        startlen,
        index: 0,
        slot: 0,
    })
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_set_iterator(obj: PyObjectRef) -> bool {
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return false;
    }
    !obj.is_null() && std::ptr::eq((*obj).ob_type, &SET_ITERATOR_TYPE)
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_set_iter_get_set(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_SetIterObject)).w_set
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_set_iter_set_set(obj: PyObjectRef, w_set: PyObjectRef) {
    (*(obj as *mut W_SetIterObject)).w_set = w_set;
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_set_iter_get_startlen(obj: PyObjectRef) -> usize {
    (*(obj as *const W_SetIterObject)).startlen
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_set_iter_set_startlen(obj: PyObjectRef, startlen: usize) {
    (*(obj as *mut W_SetIterObject)).startlen = startlen;
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_set_iter_get_index(obj: PyObjectRef) -> usize {
    (*(obj as *const W_SetIterObject)).index
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_set_iter_set_index(obj: PyObjectRef, index: usize) {
    (*(obj as *mut W_SetIterObject)).index = index;
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_set_iter_get_slot(obj: PyObjectRef) -> usize {
    (*(obj as *const W_SetIterObject)).slot
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_set_iter_set_slot(obj: PyObjectRef, slot: usize) {
    (*(obj as *mut W_SetIterObject)).slot = slot;
}

/// `setobject.py SetStrategy` class identity. A set-specific discriminant:
/// dict's `StrategyKind` names module, map, and kwargs strategies this file
/// does not have (`Ascii` is `AsciiSetStrategy`, not `UnicodeDictStrategy`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SetStrategyKind {
    Empty,
    Object,
    Bytes,
    Ascii,
    Int,
    Identity,
}

/// `setobject.py SetStrategy`. `EmptySetStrategy`, `IntegerSetStrategy`, and
/// `ObjectSetStrategy` are the live kinds; the other [`SetStrategyKind`]
/// discriminants wait for their storage boxes. Public operations read
/// `sstrategy` and `sstorage` under the same stripe lock and only call
/// [`w_set_object_storage`] for `SetStrategyKind::Object`.
pub trait SetStrategy {
    fn strategy_kind(&self) -> SetStrategyKind;
}

/// One-word strategy slot, the set-side [`crate::dictmultiobject::DictStrategyRef`].
///
/// `W_BaseSetObject.strategy` is one instance pointer. A `&dyn SetStrategy`
/// in the object would be a fat pointer, and the unit-struct singletons are
/// zero-sized, so this `#[repr(C)]` holder is what `sstrategy` stores.
#[repr(C)]
pub struct SetStrategyRef {
    /// Compared by field, the way `DictStrategyRef.kind` is, so a guard reads
    /// a `getfield` rather than a vtable call.
    pub kind: SetStrategyKind,
    pub imp: &'static dyn SetStrategy,
    /// `space.fromcache` singletons leave this null.
    pub owner: *mut u8,
}

unsafe impl Sync for SetStrategyRef {}
unsafe impl Send for SetStrategyRef {}

impl std::ops::Deref for SetStrategyRef {
    type Target = dyn SetStrategy;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.imp
    }
}

/// `setobject.py ObjectSetStrategy`. The erased box is [`SetItemsStorage`]
/// (`newset` → `r_dict(eq_w, hash_w)`).
pub struct ObjectSetStrategy;

impl SetStrategy for ObjectSetStrategy {
    fn strategy_kind(&self) -> SetStrategyKind {
        SetStrategyKind::Object
    }
}

/// `setobject.py ObjectSetStrategy` process-wide singleton
/// (`space.fromcache(ObjectSetStrategy)`).
pub static OBJECT_SET_STRATEGY: ObjectSetStrategy = ObjectSetStrategy;

/// Holder `W_SetObject.sstrategy` points at.
pub static OBJECT_SET_STRATEGY_REF: SetStrategyRef = SetStrategyRef {
    kind: SetStrategyKind::Object,
    imp: &OBJECT_SET_STRATEGY,
    owner: std::ptr::null_mut(),
};

/// `setobject.py EmptySetStrategy`. `get_empty_storage` is `erase(None)`.
pub struct EmptySetStrategy;

impl SetStrategy for EmptySetStrategy {
    fn strategy_kind(&self) -> SetStrategyKind {
        SetStrategyKind::Empty
    }
}

/// `setobject.py EmptySetStrategy` process-wide singleton
/// (`space.fromcache(EmptySetStrategy)`).
pub static EMPTY_SET_STRATEGY: EmptySetStrategy = EmptySetStrategy;

/// Holder installed by `W_BaseSetObject.switch_to_empty_strategy`.
pub static EMPTY_SET_STRATEGY_REF: SetStrategyRef = SetStrategyRef {
    kind: SetStrategyKind::Empty,
    imp: &EMPTY_SET_STRATEGY,
    owner: std::ptr::null_mut(),
};

/// `setobject.py IntegerSetStrategy`. The erased box is [`IntSetStorage`]
/// (`erase({})` of plain ints). `is_correct_type` is `is_plain_int1`.
pub struct IntegerSetStrategy;

impl SetStrategy for IntegerSetStrategy {
    fn strategy_kind(&self) -> SetStrategyKind {
        SetStrategyKind::Int
    }
}

/// `setobject.py IntegerSetStrategy` process-wide singleton
/// (`space.fromcache(IntegerSetStrategy)`).
pub static INTEGER_SET_STRATEGY: IntegerSetStrategy = IntegerSetStrategy;

/// Holder installed by `EmptySetStrategy.add` for a plain int.
pub static INTEGER_SET_STRATEGY_REF: SetStrategyRef = SetStrategyRef {
    kind: SetStrategyKind::Int,
    imp: &INTEGER_SET_STRATEGY,
    owner: std::ptr::null_mut(),
};

/// Python set object.
///
/// Layout: `[ob_header | sstorage | sstrategy | len | hash]`, the
/// `W_BaseSetObject` slots `sstorage` and `strategy` (`setobject.py`) plus the
/// atomic count and the frozenset hash cache. `sstorage` is the erased box;
/// [`SetItemsStorage`] (`ObjectSetStrategy.unerase`) or [`IntSetStorage`]
/// (`IntegerSetStrategy.unerase`).
#[repr(C)]
pub struct W_SetObject {
    pub ob_header: PyObject,
    /// `setobject.py W_BaseSetObject.sstorage`. The cast to `*mut u8` is the
    /// erase; [`w_set_object_storage`] casts it back. Null is
    /// `EmptySetStrategy.get_empty_storage` (`erase(None)`).
    pub sstorage: *mut u8,
    /// `setobject.py W_BaseSetObject.strategy`, one word. `w_set_new` stores
    /// [`EMPTY_SET_STRATEGY_REF`]; the first add stores
    /// [`INTEGER_SET_STRATEGY_REF`] or [`OBJECT_SET_STRATEGY_REF`].
    pub sstrategy: &'static SetStrategyRef,
    /// Element count, read WITHOUT the stripe lock.
    ///
    /// `Objects/setobject.c set_len` answers `len(s)` as
    /// `FT_ATOMIC_LOAD_SSIZE_RELAXED(so->used)` — a relaxed atomic load and no
    /// critical section — so a reader is entitled to a value from either side
    /// of a concurrent mutation but never to a torn one.  The JIT's `len` fold
    /// lowers to exactly that load (`set_len_descr`), which is why the slot
    /// cannot be a plain `usize`: the mutators below write it while a compiled
    /// loop is reading, and only an atomic makes that pair defined.
    ///
    /// Same size and bit validity as `usize`, so the descriptor group keeps
    /// reading it as `Type::Int` at `offset_of!` — the shape
    /// `W_TupleObject.hash` already uses.
    pub len: AtomicUsize,
    /// setobject.py `W_FrozensetObject.hash = DEFAULT_HASH`.
    pub hash: i64,
}

impl W_SetObject {
    /// `FT_ATOMIC_LOAD_SSIZE_RELAXED(so->used)`.
    #[inline]
    pub fn len_relaxed(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// `FT_ATOMIC_STORE_SSIZE_RELAXED(so->used, n)`.
    #[inline]
    pub fn set_len_relaxed(&self, n: usize) {
        self.len.store(n, Ordering::Relaxed);
    }
}

/// GC type id assigned to `W_SetObject` at JitDriver init time.
pub const W_SET_GC_TYPE_ID: u32 = 30;

/// GC-managed element table shared by `set` and `frozenset` bodies.
///
/// Keyed with [`ObjectKeyBuildHasher`](crate::dictmultiobject::ObjectKeyBuildHasher),
/// the same hasher `ObjectDictStorage` uses: `ObjectKey.hash` already *is*
/// `space.hash_w(obj)`, so the default `RandomState` would SipHash a digest
/// that is itself the hash. `rordereddict` feeds the cached integer straight
/// into the table; the multiply only spreads it into hashbrown's control bits.
pub type SetItemsStorage = crate::rordereddict::RDict<
    crate::dictmultiobject::ObjectKey,
    (),
    crate::dictmultiobject::ObjectKeyBuildHasher,
>;

/// `setobject.py IntegerSetStrategy.get_empty_dict` — `{}` of plain ints.
/// Keys are `i64` (`plain_int_w`); the table hash is [`IntKeyHash`]
/// (`ll_int_hash`), not a second digest of the key.
pub type IntSetStorage = crate::rordereddict::RDict<i64, (), crate::dictmultiobject::IntKeyHash>;

/// Runtime-assigned GC type id for the [`IntSetStorage`] entries array
/// (`GcArray` of `i64` keys and unit values; no `PyObjectRef`).
static INT_SET_ENTRIES_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for the [`IntSetStorage`] entries array.
pub fn set_int_set_entries_gc_type_id(id: u32) {
    INT_SET_ENTRIES_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for the [`IntSetStorage`] entries array.
#[majit_macros::dont_look_inside]
pub fn int_set_entries_gc_type_id() -> u32 {
    INT_SET_ENTRIES_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

impl crate::rordereddict::GcEntriesType for (i64, ()) {
    fn entries_gc_type_id() -> u32 {
        int_set_entries_gc_type_id()
    }
}

/// `setobject.py ObjectSetStrategy.unerase` — the erased `sstorage` word
/// cast back to [`SetItemsStorage`].
///
/// # Safety
/// `obj` must point to a valid `W_SetObject` whose strategy is
/// [`OBJECT_SET_STRATEGY`].
#[inline]
pub unsafe fn w_set_object_storage<'a>(obj: PyObjectRef) -> &'a SetItemsStorage {
    &*object_set_storage_ptr(&*(obj as *const W_SetObject))
}

/// Mutable [`w_set_object_storage`].
///
/// # Safety
/// Same as [`w_set_object_storage`].
#[inline]
pub unsafe fn w_set_object_storage_mut<'a>(obj: PyObjectRef) -> &'a mut SetItemsStorage {
    &mut *object_set_storage_ptr(&*(obj as *const W_SetObject))
}

/// Pointer form of [`w_set_object_storage`]. Debug-asserts `SetStrategyKind::Object`.
#[inline]
unsafe fn object_set_storage_ptr(set: &W_SetObject) -> *mut SetItemsStorage {
    debug_assert_eq!(set.sstrategy.kind, SetStrategyKind::Object);
    set.sstorage as *mut SetItemsStorage
}

/// True when `items` is still this set's live `ObjectSetStrategy` box.
///
/// `switch_to_empty_strategy` leaves `sstorage` null
/// (`EmptySetStrategy.get_empty_storage`). A probe that captured the old
/// box must not unerase the live word after that.
#[inline]
fn same_live_object_box(set: &W_SetObject, items: *mut SetItemsStorage) -> bool {
    set.sstrategy.kind == SetStrategyKind::Object && set.sstorage == items as *mut u8
}

/// `setobject.py IntegerSetStrategy.unerase`.
///
/// # Safety
/// `set` must be a live `W_SetObject` on [`INTEGER_SET_STRATEGY`].
#[inline]
unsafe fn int_set_storage_ptr(set: &W_SetObject) -> *mut IntSetStorage {
    debug_assert_eq!(set.sstrategy.kind, SetStrategyKind::Int);
    set.sstorage as *mut IntSetStorage
}

/// `IntegerSetStrategy.wrap` (`space.newint`) plus the digest `hash_w` stores
/// on an [`crate::dictmultiobject::ObjectKey`].
///
/// `intobject.py _hash_int` is that digest (`hash(1) == 1`, `hash(-1) == -2`).
/// pyre-object reaches it through `object_key_for` → `hash_w`, the same helper
/// an object-strategy probe uses, rather than a second reduction.
///
/// # Safety
/// Caller holds whatever keeps `value`'s future wrapper alive across the
/// allocation, or uses the returned key before the next collection.
unsafe fn object_key_for_plain_int(value: i64) -> crate::dictmultiobject::ObjectKey {
    let _roots = crate::gc_roots::push_roots();
    let wrapped = crate::gc_roots::pin_root(crate::w_int_new(value));
    crate::dictmultiobject::object_key_for(wrapped)
}

/// Remove the entry occupying `slot` in O(1).
///
/// `_ll_dict_del_entry` marks the index slot [`DELETED`] and clears the entry
/// in place, so every surviving key keeps both its slot number and its
/// position in the walk order.
///
/// [`DELETED`]: crate::rordereddict::DELETED
unsafe fn set_remove_slot(items: *mut SetItemsStorage, slot: usize) {
    let _ = (*items).remove_slot(slot);
}

/// The slot holding the first element at or after `from`, or `None` past the
/// last one.
///
/// A walk counts in slots, not in positions: `_ll_dict_del_entry` leaves a
/// hole where it deleted, so slot numbers outlive their neighbours' removal
/// but stop being contiguous.  A caller re-reads its key with [`w_set_key_at`]
/// at the slot this hands back and resumes from `slot + 1`.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_next_slot(obj: PyObjectRef, from: usize) -> Option<usize> {
    let _set_guard = w_set_lock(obj);
    let s = &*(obj as *const W_SetObject);
    // `setobject.py EmptyIteratorImplementation.next_entry` is always None.
    match s.sstrategy.kind {
        SetStrategyKind::Empty => None,
        // `IntegerIteratorImplementation` walks the unwrapped dict's slots.
        SetStrategyKind::Int => (*int_set_storage_ptr(s)).next_valid_slot(from),
        SetStrategyKind::Object => (*object_set_storage_ptr(s)).next_valid_slot(from),
        SetStrategyKind::Bytes | SetStrategyKind::Ascii | SetStrategyKind::Identity => None,
    }
}

// PyPy serializes set strategy/storage operations with the GIL. Pyre is
// free-threaded, so use the same narrow address-striped reentrant-lock
// adaptation as listobject and dictmultiobject. Semantic state remains on
// W_SetObject; the stripes only restore the indivisible operation boundary
// supplied by PyPy's GIL. Stripes are reset in a fork child because a vanished
// thread may have held one at fork time.
struct ForkSetLock(UnsafeCell<parking_lot::ReentrantMutex<()>>);
unsafe impl Sync for ForkSetLock {}

impl ForkSetLock {
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

static SET_LOCKS: LazyLock<Vec<ForkSetLock>> =
    LazyLock::new(|| (0..256).map(|_| ForkSetLock::new()).collect());

type SetGuard = parking_lot::lock_api::ReentrantMutexGuard<
    'static,
    parking_lot::RawMutex,
    parking_lot::RawThreadId,
    (),
>;

#[inline]
fn set_lock_index(obj: PyObjectRef) -> usize {
    (obj as usize >> 4) & (SET_LOCKS.len() - 1)
}

/// Acquire a set's stripe without letting a contending mutator prevent a GC
/// stop-the-world. Only the acquire is opaque; the guarded set operation stays
/// visible to source translation, matching listobject/dictmultiobject.
#[majit_macros::dont_look_inside]
unsafe fn w_set_lock(obj: PyObjectRef) -> SetGuard {
    let lock = SET_LOCKS[set_lock_index(obj)].get();
    if let Some(guard) = lock.try_lock() {
        return guard;
    }
    let blocked = majit_gc::gc_sync::before_external_block();
    let guard = lock.lock();
    drop(blocked);
    guard
}

/// Lock two sets in stripe order. A shared stripe is acquired once.
#[majit_macros::dont_look_inside]
unsafe fn w_set_lock_pair(left: PyObjectRef, right: PyObjectRef) -> (SetGuard, Option<SetGuard>) {
    let left_index = set_lock_index(left);
    let right_index = set_lock_index(right);
    if left_index == right_index {
        return (w_set_lock(left), None);
    }
    if left_index < right_index {
        (w_set_lock(left), Some(w_set_lock(right)))
    } else {
        (w_set_lock(right), Some(w_set_lock(left)))
    }
}

pub fn set_locks_after_fork_child() {
    for lock in SET_LOCKS.iter() {
        unsafe { lock.reinit_after_fork() };
    }
}

/// Runtime-assigned GC type id for [`SetItemsStorage`]. Like the bigint
/// payload id, this is published by `pyre-jit::eval` after the fixed-constant
/// type registrations and is never embedded in a JIT allocation descriptor.
static SET_ITEMS_GC_TYPE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for [`SetItemsStorage`].
pub fn set_set_items_gc_type_id(id: u32) {
    SET_ITEMS_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for [`SetItemsStorage`].
#[majit_macros::dont_look_inside]
pub fn set_items_gc_type_id() -> u32 {
    SET_ITEMS_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// Runtime-assigned GC type id for the [`IntSetStorage`] box.
static INT_SET_STORAGE_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for the [`IntSetStorage`] box.
pub fn set_int_set_storage_gc_type_id(id: u32) {
    INT_SET_STORAGE_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for the [`IntSetStorage`] box.
#[majit_macros::dont_look_inside]
pub fn int_set_storage_gc_type_id() -> u32 {
    INT_SET_STORAGE_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// Fixed payload size (`framework.py:811`).
pub const W_SET_OBJECT_SIZE: usize = std::mem::size_of::<W_SetObject>();

impl crate::lltype::GcType for W_SetObject {
    fn type_id() -> u32 {
        W_SET_GC_TYPE_ID
    }
    const SIZE: usize = W_SET_OBJECT_SIZE;
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_set(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &SET_TYPE) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_frozenset(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &FROZENSET_TYPE) }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_set_or_frozenset(obj: PyObjectRef) -> bool {
    unsafe { is_set(obj) || is_frozenset(obj) }
}

/// Fire the write barrier on a set whose `items` field was just replaced.
///
/// `ll_dict_setitem` records the store on the `dicttable`
/// (`rordereddict.py` `GcStruct("dicttable")`), not on the set.
/// [`set_items_write_barrier`] is that store. This one is only the
/// `sstorage` assignment (`setobject.py` `W_BaseSetObject.clear`,
/// `get_storage_copy`). A no-GC-hook fallback allocation is not
/// collector-owned.
#[inline]
fn set_write_barrier(obj: PyObjectRef) {
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
    if obj.is_null() {
        return;
    }
    let set = unsafe { &*(obj as *const W_SetObject) };
    // `EmptySetStrategy.get_empty_storage` is null. Kind and storage are
    // read together; an empty set has nothing to remember.
    if set.sstrategy.kind != SetStrategyKind::Object {
        return;
    }
    let items = unsafe { object_set_storage_ptr(set) };
    if !items.is_null() && crate::gc_hook::try_gc_owns_object(items as *mut u8) {
        crate::gc_hook::try_gc_write_barrier(items as *mut u8);
    }
}

/// `setobject.py W_BaseSetObject.switch_to_empty_strategy`.
///
/// Strategy is published before the null storage word, so a reader still
/// observing `ObjectSetStrategy` still has the previous box.
///
/// # Safety
/// `obj` must point at a valid `W_SetObject`. Caller holds `w_set_lock`.
unsafe fn switch_to_empty_strategy(obj: PyObjectRef) {
    {
        let s = &mut *(obj as *mut W_SetObject);
        s.sstrategy = &EMPTY_SET_STRATEGY_REF;
        s.sstorage = std::ptr::null_mut();
        s.set_len_relaxed(0);
        s.hash = -1;
    }
    set_write_barrier(obj);
}

/// `setobject.py EmptySetStrategy.add` — install `ObjectSetStrategy` and
/// its empty storage, then the caller performs the add. A plain int takes
/// [`switch_empty_to_int_strategy`] instead. Bytes, ascii, and identity are
/// later steps.
///
/// Storage is published before the kind, so the word is never
/// `ObjectSetStrategy` over null. `try_gc_alloc_stable_raw` does not collect;
/// the pin matches `install_empty_strategy`.
///
/// # Safety
/// `obj` must point at a valid `W_SetObject` whose strategy is
/// `EmptySetStrategy`. Caller holds `w_set_lock`.
unsafe fn switch_empty_to_object_strategy(obj: PyObjectRef) {
    let _roots = crate::gc_roots::push_roots();
    let set_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    let storage =
        crate::gc_storage::gc_alloc_storage_box(SetItemsStorage::default(), set_items_gc_type_id());
    let obj = crate::gc_roots::shadow_stack_get(set_slot);
    {
        let set = &mut *(obj as *mut W_SetObject);
        set.sstorage = storage as *mut u8;
        set.sstrategy = &OBJECT_SET_STRATEGY_REF;
    }
    set_write_barrier(obj);
}

/// `setobject.py EmptySetStrategy.add` — `is_plain_int1` installs
/// `IntegerSetStrategy` and `get_empty_storage` (`erase({})`).
///
/// # Safety
/// `obj` must point at a valid `W_SetObject` whose strategy is
/// `EmptySetStrategy`. Caller holds `w_set_lock`.
unsafe fn switch_empty_to_int_strategy(obj: PyObjectRef) {
    let _roots = crate::gc_roots::push_roots();
    let set_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    let storage =
        crate::gc_storage::gc_alloc_storage_box(IntSetStorage::new(), int_set_storage_gc_type_id());
    let obj = crate::gc_roots::shadow_stack_get(set_slot);
    {
        let set = &mut *(obj as *mut W_SetObject);
        set.sstorage = storage as *mut u8;
        set.sstrategy = &INTEGER_SET_STRATEGY_REF;
    }
    set_write_barrier(obj);
}

/// `setobject.py W_BaseSetObject.switch_to_object_strategy` for
/// `IntegerSetStrategy`: `getdict_w` wraps each key with `newint`, then
/// `ObjectSetStrategy.erase` installs that dict.
///
/// Slot numbers are preserved, tombstones included
/// ([`crate::rordereddict::RDict::map_keys_preserving_layout`]), because
/// `W_SetIterObject.slot` indexes the live table across the switch.
/// The elements do not change, so the frozenset hash cache is left alone.
///
/// # Safety
/// `obj` must point at a valid `W_SetObject` on [`INTEGER_SET_STRATEGY`].
/// Caller holds `w_set_lock`. `obj` must already be rooted: `w_int_new` collects.
unsafe fn switch_int_to_object_strategy(obj: PyObjectRef) {
    let _roots = crate::gc_roots::push_roots();
    let set_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    let n = {
        let set = &*(crate::gc_roots::shadow_stack_get(set_slot) as *const W_SetObject);
        (*int_set_storage_ptr(set)).entry_slots()
    };
    let mut raw: Vec<Option<i64>> = Vec::with_capacity(n);
    {
        let set = &*(crate::gc_roots::shadow_stack_get(set_slot) as *const W_SetObject);
        let old = int_set_storage_ptr(set);
        for slot in 0..n {
            raw.push((*old).get_slot(slot).map(|(key, _)| *key));
        }
    }
    let mut hashes: Vec<i64> = Vec::with_capacity(n);
    let live_base = crate::gc_roots::shadow_stack_len();
    for slot in 0..n {
        if let Some(key) = raw[slot] {
            let wrapped = crate::gc_roots::pin_root(crate::w_int_new(key));
            let keyed = crate::dictmultiobject::object_key_for(wrapped);
            hashes.push(keyed.hash);
        }
    }
    let mut slot_keys = Vec::with_capacity(n);
    let mut live = 0usize;
    for slot in 0..n {
        if raw[slot].is_some() {
            slot_keys.push(crate::dictmultiobject::ObjectKey {
                hash: hashes[live],
                obj: crate::gc_roots::shadow_stack_get(live_base + live),
            });
            live += 1;
        } else {
            slot_keys.push(crate::dictmultiobject::ObjectKey {
                hash: 0,
                obj: std::ptr::null_mut(),
            });
        }
    }
    let mapped = {
        let set = &*(crate::gc_roots::shadow_stack_get(set_slot) as *const W_SetObject);
        (*int_set_storage_ptr(set)).map_keys_preserving_layout(&slot_keys)
    };
    // `try_gc_alloc_stable_raw` does not collect. The wrapped keys stay on
    // the shadow stack until the new box, which traces them, is installed.
    let storage = crate::gc_storage::gc_alloc_storage_box(mapped, set_items_gc_type_id());
    let obj = crate::gc_roots::shadow_stack_get(set_slot);
    {
        let set = &mut *(obj as *mut W_SetObject);
        set.sstorage = storage as *mut u8;
        set.sstrategy = &OBJECT_SET_STRATEGY_REF;
    }
    set_write_barrier(obj);
    set_items_write_barrier(storage);
}

/// Publish [`IntSetStorage`]'s length when the set is still on
/// `IntegerSetStrategy`.
///
/// # Safety
/// `obj` must point at a valid `W_SetObject`. Caller holds `w_set_lock`.
unsafe fn publish_int_len(obj: PyObjectRef) {
    let set = &mut *(obj as *mut W_SetObject);
    if set.sstrategy.kind != SetStrategyKind::Int {
        return;
    }
    set.set_len_relaxed((*int_set_storage_ptr(set)).len());
    set.hash = -1;
}

/// `W_SetObject._discard_from_set` for an int set: length 0 calls
/// `switch_to_empty_strategy`.
///
/// # Safety
/// `obj` must point at a valid `W_SetObject`. Caller holds `w_set_lock`.
unsafe fn publish_int_discard(obj: PyObjectRef) {
    let emptied = {
        let set = &*(obj as *const W_SetObject);
        set.sstrategy.kind == SetStrategyKind::Int && (*int_set_storage_ptr(set)).len() == 0
    };
    if emptied {
        switch_to_empty_strategy(obj);
    } else {
        publish_int_len(obj);
    }
}

/// `AbstractUnwrappedSetStrategy.add` when `is_correct_type`: `d[unwrap] = None`.
///
/// The wrong-type arm is the caller's `switch_to_object_strategy`.
///
/// # Safety
/// `obj` must point at a valid `W_SetObject` on [`INTEGER_SET_STRATEGY`],
/// and `key.obj` must be `is_plain_int1`. Caller holds `w_set_lock`.
unsafe fn int_set_add_unwrapped(
    obj: PyObjectRef,
    key: crate::dictmultiobject::ObjectKey,
) -> Result<(), SetUpdateError> {
    let unwrapped = crate::listobject::plain_int_w(key.obj);
    let inserted = {
        let set = &mut *(obj as *mut W_SetObject);
        (*int_set_storage_ptr(set)).insert(unwrapped, ()).is_none()
    };
    if inserted {
        publish_int_len(obj);
    }
    Ok(())
}

/// `AbstractUnwrappedSetStrategy.has_key` when `is_correct_type`.
///
/// # Safety
/// `obj` must point at a valid `W_SetObject` on [`INTEGER_SET_STRATEGY`],
/// and `key.obj` must be `is_plain_int1`. Caller holds `w_set_lock`.
unsafe fn int_set_contains_unwrapped(
    obj: PyObjectRef,
    key: crate::dictmultiobject::ObjectKey,
) -> bool {
    let unwrapped = crate::listobject::plain_int_w(key.obj);
    let set = &*(obj as *const W_SetObject);
    (*int_set_storage_ptr(set)).contains_key(&unwrapped)
}

/// `AbstractUnwrappedSetStrategy.remove` when `is_correct_type`.
///
/// `to_empty` is `W_SetObject._discard_from_set` (switch when the set
/// becomes empty). `delitem_with_hash` leaves an empty int dict in place.
///
/// # Safety
/// `obj` must point at a valid `W_SetObject` on [`INTEGER_SET_STRATEGY`],
/// and `key.obj` must be `is_plain_int1`. Caller holds `w_set_lock`.
unsafe fn int_set_remove_unwrapped(
    obj: PyObjectRef,
    key: crate::dictmultiobject::ObjectKey,
    to_empty: bool,
) -> bool {
    let unwrapped = crate::listobject::plain_int_w(key.obj);
    let removed = {
        let set = &mut *(obj as *mut W_SetObject);
        (*int_set_storage_ptr(set)).remove(&unwrapped).is_some()
    };
    if removed {
        if to_empty {
            publish_int_discard(obj);
        } else {
            publish_int_len(obj);
        }
    }
    removed
}

/// `EmptySetStrategy.add` / `AbstractUnwrappedSetStrategy.add`: promote an
/// empty set, or switch an int set that was handed a non-int.
///
/// `hash` is the digest the caller already took (`hash_w`). The returned
/// key's object is reloaded after any allocation. `obj_slot` / `key_slot`
/// are shadow-stack indexes the caller pinned.
///
/// # Safety
/// Caller holds `w_set_lock`. Both slots name the set and the key.
unsafe fn prepare_set_for_key(
    obj_slot: usize,
    key_slot: usize,
    hash: i64,
) -> crate::dictmultiobject::ObjectKey {
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let key_obj = crate::gc_roots::shadow_stack_get(key_slot);
    let kind = (*(obj as *const W_SetObject)).sstrategy.kind;
    if kind == SetStrategyKind::Empty {
        if crate::listobject::is_plain_int1(key_obj) {
            switch_empty_to_int_strategy(obj);
        } else {
            switch_empty_to_object_strategy(obj);
        }
    } else if kind == SetStrategyKind::Int && !crate::listobject::is_plain_int1(key_obj) {
        switch_int_to_object_strategy(obj);
    }
    crate::dictmultiobject::ObjectKey {
        hash,
        obj: crate::gc_roots::shadow_stack_get(key_slot),
    }
}

/// Publish `items`'s length when that box is still `dst`'s live storage.
///
/// A re-entrant `clear` has already stored length 0 on the empty strategy;
/// copying the orphan's length back would resurrect it.
///
/// # Safety
/// `dst` must point at a valid `W_SetObject`. `items` must be a live
/// [`SetItemsStorage`] box when it is still `dst`'s storage.
unsafe fn publish_len_if_live_box(dst: PyObjectRef, items: *mut SetItemsStorage) {
    let set = &mut *(dst as *mut W_SetObject);
    if same_live_object_box(set, items) {
        set.set_len_relaxed((*items).len());
        set.hash = -1;
    }
}

/// `setobject.py W_SetObject._discard_from_set`: a removal that leaves
/// `length() == 0` calls `switch_to_empty_strategy`. A removal against a box
/// `clear` already detached does not touch the live set.
///
/// # Safety
/// `obj` must point at a valid `W_SetObject`. `items` must be a live
/// [`SetItemsStorage`] box when it is still `obj`'s storage. Caller holds
/// `w_set_lock`.
unsafe fn publish_discard_if_live_box(obj: PyObjectRef, items: *mut SetItemsStorage) {
    let emptied = {
        let set = &mut *(obj as *mut W_SetObject);
        if !same_live_object_box(set, items) {
            false
        } else {
            let new_len = (*items).len();
            if new_len == 0 {
                true
            } else {
                set.set_len_relaxed(new_len);
                set.hash = -1;
                false
            }
        }
    };
    if emptied {
        switch_to_empty_strategy(obj);
    }
}

/// Remember `items` after a store of a GC reference into that box.
///
/// The box is old (`gc_alloc_storage_box`) and `set_items_storage_custom_trace`
/// walks its `ObjectKey.obj` slots. A minor reaches an old box only from the
/// remembered set — `drag_out` does not trace an old root — including a box
/// `clear` has already detached (`d = self.unerase(w_set.sstorage)`,
/// `setobject.py`).
#[inline]
fn set_items_write_barrier(items: *mut SetItemsStorage) {
    crate::gc_hook::try_gc_write_barrier(items as *mut u8);
}

/// Allocate an empty `set`.
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`), the
/// `w_dict_new` twin: the body is a nursery allocation
/// (`try_gc_alloc_nursery_raw`). Residualising the constructor models it by
/// signature — a plain `PyObjectRef` GCREF. Storage is
/// `EmptySetStrategy.get_empty_storage` (`erase(None)`); the `SetItemsStorage`
/// box appears on the first add.
#[majit_macros::dont_look_inside]
pub fn w_set_new() -> PyObjectRef {
    alloc_set_object(&SET_TYPE)
}

/// Allocate an empty `frozenset`.
///
/// Same body as [`w_set_new`] with the constant `&FROZENSET_TYPE` baked
/// into `ob_type`. `#[dont_look_inside]` for the same nursery-allocation
/// reason as [`w_set_new`].
#[majit_macros::dont_look_inside]
pub fn w_frozenset_new() -> PyObjectRef {
    alloc_set_object(&FROZENSET_TYPE)
}

fn alloc_set_object(set_type: &'static PyType) -> PyObjectRef {
    // Allocate the body on the same nursery bump as `w_tuple_new`
    // (`malloc_fixedsize`). Falls back to `malloc_typed` when no GC hook is
    // installed (unit tests). `EmptySetStrategy.get_empty_storage` is
    // `erase(None)`, so there is no items box to pin. The class is reloaded
    // after the nursery allocation (`get_instantiate` / the body malloc can
    // collect).
    let _roots = crate::gc_roots::push_roots();
    let class_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(get_instantiate(set_type));
    let raw = crate::gc_hook::try_gc_alloc_nursery_raw(W_SET_GC_TYPE_ID, W_SET_OBJECT_SIZE);
    let body = W_SetObject {
        ob_header: PyObject {
            ob_type: set_type as *const PyType,
            w_class: crate::gc_roots::shadow_stack_get(class_slot),
        },
        sstorage: std::ptr::null_mut(),
        sstrategy: &EMPTY_SET_STRATEGY_REF,
        len: crate::object_array::length_cell(0),
        hash: -1,
    };
    if !raw.is_null() {
        unsafe {
            std::ptr::write(raw as *mut W_SetObject, body);
        }
        crate::gc_hook::try_gc_write_barrier_managed(raw);
        raw as PyObjectRef
    } else {
        crate::lltype::malloc_typed(body) as PyObjectRef
    }
}

/// Allocate a populated set-like from a slice of elements (deduped).
///
/// A `&[PyObjectRef]` is not a root area, so the caller's words go stale at
/// the first insert: `w_set_add` hashes with `space.hash_w`, a collection
/// point that may run a user `__hash__`, and the set's own table can grow.
/// The whole slice is therefore published as one livevar set before the
/// constructor allocates, and both the set and each element are read back
/// out of their slots per insert.
fn set_from_items(new_set: impl FnOnce() -> PyObjectRef, items: &[PyObjectRef]) -> PyObjectRef {
    let roots = crate::gc_roots::push_roots();
    let base = roots.publish(items);
    roots.normalize(base, items.len());
    let set = base + items.len();
    let _ = roots.pin_root(new_set());
    for i in 0..items.len() {
        unsafe { w_set_add(roots.get(set), roots.get(base + i)) };
    }
    roots.get(set)
}

/// Allocate a populated set from a slice of elements (deduped).
pub fn w_set_from_items(items: &[PyObjectRef]) -> PyObjectRef {
    set_from_items(w_set_new, items)
}

/// Allocate a populated frozenset from a slice of elements (deduped).
pub fn w_frozenset_from_items(items: &[PyObjectRef]) -> PyObjectRef {
    set_from_items(w_frozenset_new, items)
}

/// Insert an element. No-op when already present.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_add(obj: PyObjectRef, item: PyObjectRef) {
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::pin_roots(&[obj, item]);
    let item_slot = obj_slot + 1;
    let key = crate::dictmultiobject::object_key_for(crate::gc_roots::shadow_stack_get(item_slot));
    let key = crate::dictmultiobject::object_key_hashed(
        crate::gc_roots::shadow_stack_get(item_slot),
        key.hash,
    );
    let _ = w_set_insert_key_checked(crate::gc_roots::shadow_stack_get(obj_slot), key);
}

/// Insert an element keyed on a `space.hash_w` digest the caller already
/// holds, propagating an `eq_w` raise from the bucket probe.
///
/// `setobject.py newset` builds the backing `r_dict` with both
/// `space.eq_w` and `space.hash_w`, so one `add` hashes the element once and
/// compares it with `eq_w`, and either callback raising aborts the store.
/// A user `__hash__` is a collection point that can move both `obj` and
/// `item`, so the hash is taken by the caller while they are still rooted and
/// the digest handed down here; `hash` must be the `space.hash_w` result for
/// `item` (see [`object_key_hashed`](crate::dictmultiobject::object_key_hashed)).
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_add_hashed_checked(
    obj: PyObjectRef,
    item: PyObjectRef,
    hash: i64,
) -> Result<(), SetUpdateError> {
    w_set_insert_key_checked(obj, crate::dictmultiobject::object_key_hashed(item, hash))
}

/// Run a set operation with the callback-free guard raised, so
/// `dict_keys_equal` answers from its builtin type ladder and no user
/// `__eq__` observes or mutates the set mid-probe.
///
/// Returns `None` when a comparison escaped the ladder: `op` withholds its
/// mutation in that case, so the set is untouched and the caller re-runs the
/// operation by scanning entries without holding a table borrow across a
/// callback.
#[inline]
unsafe fn callback_free_set_op<T>(
    op: impl FnOnce() -> T,
) -> Option<Result<T, crate::dictmultiobject::DictKeyError>> {
    crate::dict_eq_hook::begin_callback_free_probe();
    let result = op();
    if crate::dict_eq_hook::end_callback_free_probe() {
        return None;
    }
    if crate::dictmultiobject::take_dict_key_error() {
        return Some(Err(crate::dictmultiobject::DictKeyError));
    }
    Some(Ok(result))
}

/// Find `key` by scanning same-hash entries of a *captured* storage box one at
/// a time.  `items` is the box the caller snapshotted at operation entry and
/// pinned, mirroring `AbstractUnwrappedSetStrategy`'s `d =
/// self.unerase(w_set.sstorage)` (`setobject.py:934`): the whole probe runs
/// against that box, so if a probing `__eq__` swaps the set's live storage
/// (`AbstractUnwrappedSetStrategy.clear` → `switch_to_empty_strategy`) the
/// scan completes against the now-orphaned snapshot. The live word is
/// `EmptySetStrategy.get_empty_storage` (`erase(None)`), not a new table.
///
/// The storage box has a stable address (`gc_alloc_storage_box` →
/// `try_gc_alloc_stable_raw`), so `items` never moves; the caller's pin only
/// keeps an orphaned box alive across the callbacks.  The table borrow ends
/// before equality can call user code; a callback that grows or reorders the
/// captured box restarts the scan (`ll_dict_lookup` paranoia,
/// `rordereddict.py`).  The generation counter is exact here, not a
/// proxy: it is bumped by every compaction and reindex, the only two things
/// that move an entry out from under a slot number.
unsafe fn scan_set_key_reentrant(
    items: *mut SetItemsStorage,
    mut key: crate::dictmultiobject::ObjectKey,
) -> Result<(Option<usize>, crate::dictmultiobject::ObjectKey), crate::dictmultiobject::DictKeyError>
{
    'restart: loop {
        let generation = (*items).generation();
        let mut i = 0;
        loop {
            let Some((slot, stored_hash, stored_obj)) = (*items)
                .next_entry(i)
                .map(|(slot, stored, _)| (slot, stored.hash, stored.obj))
            else {
                return Ok((None, key));
            };

            if stored_hash == key.hash {
                let _roots = crate::gc_roots::push_roots();
                let stored_slot = crate::gc_roots::shadow_stack_len();
                let stored_obj = crate::gc_roots::pin_root(stored_obj);
                let key_slot = crate::gc_roots::shadow_stack_len();
                key.obj = crate::gc_roots::pin_root(key.obj);

                let equal = crate::dictmultiobject::dict_keys_equal(stored_obj, key.obj);
                let stored_obj = crate::gc_roots::shadow_stack_get(stored_slot);
                key.obj = crate::gc_roots::shadow_stack_get(key_slot);
                if crate::dictmultiobject::take_dict_key_error() {
                    return Err(crate::dictmultiobject::DictKeyError);
                }
                // Validate the paranoia condition before acting on the result:
                // `ll_dict_lookup` restarts even when the comparison answered
                // `true`, because a callback that reallocated the buffer or moved
                // the candidate leaves the matched index stale
                // (`rordereddict.py`).
                let disturbed = (*items).generation() != generation
                    // `entries.valid(index) && entries[index].key == checkingkey`.
                    || !(*items).get_slot(slot).is_some_and(|(stored, _)| {
                        stored.hash == stored_hash && stored.obj == stored_obj
                    });
                if disturbed {
                    continue 'restart;
                }
                if equal {
                    return Ok((Some(slot), key));
                }
            }
            i = slot + 1;
        }
    }
}

/// Snapshot the set's storage box and pin it for a reentrant probe, mirroring
/// PyPy's capture-before-probe (`d = self.unerase(w_set.sstorage)`).  The pin
/// keeps the box alive even if a probing `__eq__` swaps the set's live storage
/// (`w_set_clear`); the returned pointer is used for the whole operation.
#[inline]
unsafe fn capture_set_items(obj: PyObjectRef) -> *mut SetItemsStorage {
    let items = object_set_storage_ptr(&*(obj as *const W_SetObject));
    let _ = crate::gc_roots::pin_root(items as PyObjectRef);
    items
}

/// Store a key that carries its own digest, propagating an `eq_w` raise from
/// the bucket probe.
///
/// `setobject.py _intersect_unwrapped` places a key it took from another
/// set with `setitem_with_hash(result, key, keyhash, None)`, i.e. under the
/// digest the key already carries rather than one taken afresh.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`, and `key.hash` must be the
/// `space.hash_w` digest of `key.obj`.
pub unsafe fn w_set_insert_key_checked(
    obj: PyObjectRef,
    key: crate::dictmultiobject::ObjectKey,
) -> Result<(), SetUpdateError> {
    let _set_guard = w_set_lock(obj);
    // `setobject.py EmptySetStrategy.add` picks from the key
    // (`is_plain_int1` → `IntegerSetStrategy`, else `ObjectSetStrategy`)
    // and `AbstractUnwrappedSetStrategy.add` switches an int set that is
    // handed any other key. Both run before the store.
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    let key_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(key.obj);
    let key = prepare_set_for_key(obj_slot, key_slot, key.hash);
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    if (*(obj as *const W_SetObject)).sstrategy.kind == SetStrategyKind::Int {
        return int_set_add_unwrapped(obj, key);
    }
    w_set_insert_key_reentrant(obj, key)
}

/// Membership test for a key that carries its own digest, propagating an
/// `eq_w` raise from the bucket probe.
///
/// `setobject.py _intersect_unwrapped` probes the other side with
/// `contains_with_hash(d_other, key, keyhash)`, reusing the digest the key was
/// stored under instead of hashing it again.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`, and `key.hash` must be the
/// `space.hash_w` digest of `key.obj`.
pub unsafe fn w_set_contains_key_checked(
    obj: PyObjectRef,
    key: crate::dictmultiobject::ObjectKey,
) -> Result<bool, crate::dictmultiobject::DictKeyError> {
    let _set_guard = w_set_lock(obj);
    // `setobject.py EmptySetStrategy.has_key` hashes (the caller already
    // did, building `key`) and then returns False.
    if (*(obj as *const W_SetObject)).sstrategy.kind == SetStrategyKind::Empty {
        return Ok(false);
    }
    // `AbstractUnwrappedSetStrategy.has_key` switches on the wrong type
    // before the object probe. Pin across that allocation.
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    let key_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(key.obj);
    if (*(crate::gc_roots::shadow_stack_get(obj_slot) as *const W_SetObject))
        .sstrategy
        .kind
        == SetStrategyKind::Int
    {
        let key_obj = crate::gc_roots::shadow_stack_get(key_slot);
        if crate::listobject::is_plain_int1(key_obj) {
            return Ok(int_set_contains_unwrapped(
                crate::gc_roots::shadow_stack_get(obj_slot),
                crate::dictmultiobject::ObjectKey {
                    hash: key.hash,
                    obj: key_obj,
                },
            ));
        }
        switch_int_to_object_strategy(crate::gc_roots::shadow_stack_get(obj_slot));
    }
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let key = crate::dictmultiobject::ObjectKey {
        hash: key.hash,
        obj: crate::gc_roots::shadow_stack_get(key_slot),
    };
    if let Some(result) = callback_free_set_op(|| {
        let s = &*(obj as *const W_SetObject);
        (*object_set_storage_ptr(s)).contains_key(&key)
    }) {
        return result;
    }
    let items = capture_set_items(obj);
    let (found, _) = scan_set_key_reentrant(items, key)?;
    Ok(found.is_some())
}

/// Remove a key that carries its own digest, propagating an `eq_w` raise from
/// the bucket probe. Returns true when an element was removed.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`, and `key.hash` must be the
/// `space.hash_w` digest of `key.obj`.
pub unsafe fn w_set_discard_key_checked(
    obj: PyObjectRef,
    key: crate::dictmultiobject::ObjectKey,
) -> Result<bool, crate::dictmultiobject::DictKeyError> {
    let _set_guard = w_set_lock(obj);
    // `setobject.py EmptySetStrategy.remove` returns False. The caller
    // hashed while building `key`, so an unhashable key has already raised.
    if (*(obj as *const W_SetObject)).sstrategy.kind == SetStrategyKind::Empty {
        return Ok(false);
    }
    // `AbstractUnwrappedSetStrategy.remove` switches on the wrong type.
    // `_discard_from_set` then returns the set to `EmptySetStrategy`.
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(obj);
    let key_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(key.obj);
    if (*(crate::gc_roots::shadow_stack_get(obj_slot) as *const W_SetObject))
        .sstrategy
        .kind
        == SetStrategyKind::Int
    {
        let key_obj = crate::gc_roots::shadow_stack_get(key_slot);
        if crate::listobject::is_plain_int1(key_obj) {
            return Ok(int_set_remove_unwrapped(
                crate::gc_roots::shadow_stack_get(obj_slot),
                crate::dictmultiobject::ObjectKey {
                    hash: key.hash,
                    obj: key_obj,
                },
                true,
            ));
        }
        switch_int_to_object_strategy(crate::gc_roots::shadow_stack_get(obj_slot));
    }
    let obj = crate::gc_roots::shadow_stack_get(obj_slot);
    let key = crate::dictmultiobject::ObjectKey {
        hash: key.hash,
        obj: crate::gc_roots::shadow_stack_get(key_slot),
    };
    if let Some(result) = callback_free_set_op(|| {
        let items = {
            let s = &*(obj as *const W_SetObject);
            object_set_storage_ptr(s)
        };
        let index = (*items).index_of(&key);
        if crate::dict_eq_hook::callback_free_probe_broken() {
            return false;
        }
        let Some(index) = index else {
            return false;
        };
        set_remove_slot(items, index);
        publish_discard_if_live_box(obj, items);
        true
    }) {
        return result;
    }

    let items = capture_set_items(obj);
    let (found, _) = scan_set_key_reentrant(items, key)?;
    if let Some(index) = found {
        // Remove from the captured box; a `clear` during the probe orphans it,
        // leaving the live storage untouched (`discard` of an absent element).
        // A removal that empties the live box runs `switch_to_empty_strategy`
        // (`W_SetObject._discard_from_set`).
        set_remove_slot(items, index);
        let obj = crate::gc_roots::shadow_stack_get(obj_slot);
        publish_discard_if_live_box(obj, items);
        return Ok(true);
    }
    Ok(false)
}

/// Membership test.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_contains(obj: PyObjectRef, item: PyObjectRef) -> bool {
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::pin_roots(&[obj, item]);
    let item_slot = obj_slot + 1;
    let key = crate::dictmultiobject::object_key_for(crate::gc_roots::shadow_stack_get(item_slot));
    w_set_contains_key_checked(crate::gc_roots::shadow_stack_get(obj_slot), key).unwrap_or(false)
}

/// Fallible variant of [`w_set_contains`].
///
/// `setobject.py EmptySetStrategy.has_key` — the element is hashed
/// even when the set is empty, so an unhashable element raises rather than
/// reading as absent.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_contains_checked(
    obj: PyObjectRef,
    item: PyObjectRef,
) -> Result<bool, crate::dictmultiobject::DictKeyError> {
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::pin_roots(&[obj, item]);
    let item_slot = obj_slot + 1;
    let key = crate::dictmultiobject::object_key_for_checked(crate::gc_roots::shadow_stack_get(
        item_slot,
    ))?;
    w_set_contains_key_checked(crate::gc_roots::shadow_stack_get(obj_slot), key)
}

/// Remove an element if present. Returns true when removed.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_discard(obj: PyObjectRef, item: PyObjectRef) -> bool {
    // Hashing the element can run `__hash__` and collect: the set and the
    // element are livevars across it, as in `w_set_contains`.
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::pin_roots(&[obj, item]);
    let item_slot = obj_slot + 1;
    let key = crate::dictmultiobject::object_key_for(crate::gc_roots::shadow_stack_get(item_slot));
    w_set_discard_key_checked(crate::gc_roots::shadow_stack_get(obj_slot), key).unwrap_or(false)
}

/// Fallible variant of [`w_set_discard`].
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_discard_checked(
    obj: PyObjectRef,
    item: PyObjectRef,
) -> Result<bool, crate::dictmultiobject::DictKeyError> {
    // Same livevar set as [`w_set_discard`].
    let _roots = crate::gc_roots::push_roots();
    let obj_slot = crate::gc_roots::pin_roots(&[obj, item]);
    let item_slot = obj_slot + 1;
    let key = crate::dictmultiobject::object_key_for_checked(crate::gc_roots::shadow_stack_get(
        item_slot,
    ))?;
    w_set_discard_key_checked(crate::gc_roots::shadow_stack_get(obj_slot), key)
}

/// Remove every element.
///
/// `setobject.py W_BaseSetObject.clear` → `AbstractUnwrappedSetStrategy.clear`
/// → `switch_to_empty_strategy`. `EmptySetStrategy.clear` is a no-op.
/// `get_empty_storage` is `erase(None)`, so the live word becomes null rather
/// than an emptied box. A probe that captured the old box
/// (`scan_set_key_reentrant`) keeps running against that orphan: a later
/// insert lands in the dropped box and is lost, and a membership test
/// completes against the snapshot.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_clear(obj: PyObjectRef) {
    let _set_guard = w_set_lock(obj);
    if (*(obj as *const W_SetObject)).sstrategy.kind == SetStrategyKind::Empty {
        return;
    }
    switch_to_empty_strategy(obj);
}

/// Remove and return an arbitrary stored element without hashing it again.
///
/// `setobject.py ObjectSetStrategy.popitem` delegates to the backing
/// dictionary's `popitem`; the key already occupies a bucket, so no user
/// `__hash__` or `__eq__` callback occurs while it leaves the set.
///
/// # Safety
/// `obj` must point to a valid mutable `W_SetObject`.
pub unsafe fn w_set_popitem(obj: PyObjectRef) -> Option<PyObjectRef> {
    let _set_guard = w_set_lock(obj);
    let kind = (*(obj as *const W_SetObject)).sstrategy.kind;
    // `setobject.py EmptySetStrategy.popitem` raises KeyError. The caller
    // turns `None` into that error. `AbstractUnwrappedSetStrategy.popitem`
    // does not call `switch_to_empty_strategy` when the last element leaves,
    // so the set stays on its strategy with an empty dict.
    if kind == SetStrategyKind::Empty {
        return None;
    }
    if kind == SetStrategyKind::Int {
        // `IntegerSetStrategy.popitem` → `wrap` (`space.newint`).
        let raw = {
            let s = &mut *(obj as *mut W_SetObject);
            let entries = &mut *int_set_storage_ptr(s);
            let (key, ()) = entries.pop()?;
            s.set_len_relaxed(s.len_relaxed() - 1);
            s.hash = -1;
            key
        };
        let _roots = crate::gc_roots::push_roots();
        let _ = crate::gc_roots::pin_root(obj);
        return Some(crate::w_int_new(raw));
    }
    let s = &mut *(obj as *mut W_SetObject);
    let entries = &mut *object_set_storage_ptr(s);
    let (key, ()) = entries.pop()?;
    s.set_len_relaxed(s.len_relaxed() - 1);
    s.hash = -1;
    Some(key.obj)
}

/// Take over a copy of another set's storage, keeping the digest each element
/// was stored under.
///
/// `setobject.py:875` assigns a fresh storage table to the GC pointer field
/// (`w_set.sstorage = w_other.get_storage_copy()`), while
/// `ObjectSetStrategy.get_storage_copy` (`setobject.py`) creates that table
/// with `self.erase(d.copy())`. PyPy's underlying table is the GC-managed
/// `rdict.py:210` `GcStruct("dicttable")`. Do the same field reassignment here,
/// rather than overwriting the old table's pointee. Copying the buckets is
/// what makes the operand's elements reach the new set without handing them
/// to a user `__hash__` (or `__eq__`) a second time.
///
/// Cloning does not call back into user code, and the storage box uses the
/// non-collecting stable old-generation allocator, so there is no collection
/// point between reading `src`'s table and installing the new field value.
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`), the
/// `w_set_new` twin: cloning `src`'s `SetItemsStorage` (`RDict<ObjectKey,
/// ()>`) and boxing it into `d.sstorage` is a foreign `RDict::clone` +
/// storage-box write. Tracing into it carries that host container op into the
/// caller and unifies the `sstorage` field as `Instance(RDict)`; residualising
/// the whole assignment keeps the box off the trace, modelling it as a void
/// effect on two GCREFs.
///
/// # Safety
/// `dst` and `src` must point to valid `W_SetObject`s.
#[majit_macros::dont_look_inside]
pub unsafe fn w_set_copy_storage_from(dst: PyObjectRef, src: PyObjectRef) {
    let (_first_guard, _second_guard) = w_set_lock_pair(dst, src);
    // `setobject.py EmptySetStrategy.get_storage_copy` returns `sstorage`
    // unchanged (`erase(None)`). `copy_real` installs that with
    // `EmptySetStrategy`. `EmptySetStrategy.update` steals the same pair.
    let src_kind = (*(src as *const W_SetObject)).sstrategy.kind;
    if src_kind == SetStrategyKind::Empty {
        switch_to_empty_strategy(dst);
        return;
    }
    // `IntegerSetStrategy.get_storage_copy` is `erase(d.copy())` under the
    // same strategy. The keys are `i64`, so there is no element barrier.
    if src_kind == SetStrategyKind::Int {
        let copied = (*int_set_storage_ptr(&*(src as *const W_SetObject))).clone();
        let len = copied.len();
        {
            let d = &mut *(dst as *mut W_SetObject);
            d.sstorage =
                crate::gc_storage::gc_alloc_storage_box(copied, int_set_storage_gc_type_id())
                    as *mut u8;
            d.sstrategy = &INTEGER_SET_STRATEGY_REF;
            d.set_len_relaxed(len);
            d.hash = -1;
        }
        set_write_barrier(dst);
        return;
    }
    let copied = (*object_set_storage_ptr(&*(src as *const W_SetObject))).clone();
    // `gc_alloc_storage_box` is a stable allocation and never collects.
    {
        let d = &mut *(dst as *mut W_SetObject);
        // Box first, then the kind, so the word is never Object over null.
        d.sstorage =
            crate::gc_storage::gc_alloc_storage_box(copied, set_items_gc_type_id()) as *mut u8;
        d.sstrategy = &OBJECT_SET_STRATEGY_REF;
        d.set_len_relaxed((*object_set_storage_ptr(d)).len());
        d.hash = -1;
    }
    // `sstorage` assignment on the set, then the copied keys on the new table.
    // The clone filled a host `RDict` before the box existed, so the element
    // barrier did not run for those stores.
    set_write_barrier(dst);
    let d = &*(dst as *const W_SetObject);
    if (*object_set_storage_ptr(d)).len() != 0 {
        set_items_write_barrier(object_set_storage_ptr(d));
    }
}

/// Remove a set operand's elements, keeping the digests it holds.
///
/// `setobject.py _difference_update_unwrapped` — the operand's keys
/// are deleted out of self under the digests they already carry
/// (`delitem_with_hash`), and a missing one is not an error.
///
/// `:1032-1034` gives the two sides sharing one storage its own branch: that is
/// `s -= s`, which empties self. It also cannot be done by the walk below —
/// removing renumbers the very storage being walked, so every second element
/// would be stepped over.
///
/// Both sides `IntegerSetStrategy`: `_difference_unwrapped` /
/// `_difference_update_unwrapped` on the `i64` tables. No `eq_w`.
///
/// # Safety
/// Both sets are on [`INTEGER_SET_STRATEGY`] and their boxes differ.
/// Caller holds `w_set_lock_pair`. `dst_slot` and `src_slot` are pinned.
unsafe fn int_difference_update(dst_slot: usize, src_slot: usize) -> Result<(), SetUpdateError> {
    let dst_len = w_set_len(crate::gc_roots::shadow_stack_get(dst_slot));
    let src_len = w_set_len(crate::gc_roots::shadow_stack_get(src_slot));
    if dst_len < src_len {
        let mut keep = Vec::new();
        {
            let dst = crate::gc_roots::shadow_stack_get(dst_slot);
            let src = crate::gc_roots::shadow_stack_get(src_slot);
            let dst_storage = int_set_storage_ptr(&*(dst as *const W_SetObject));
            let src_storage = int_set_storage_ptr(&*(src as *const W_SetObject));
            let mut next = 0;
            while let Some(slot) = (*dst_storage).next_valid_slot(next) {
                let key = *(*dst_storage).get_slot(slot).unwrap().0;
                if !(*src_storage).contains_key(&key) {
                    keep.push(key);
                }
                next = slot + 1;
            }
        }
        let mut fresh = IntSetStorage::new();
        for key in &keep {
            fresh.insert(*key, ());
        }
        let len = fresh.len();
        let storage = crate::gc_storage::gc_alloc_storage_box(fresh, int_set_storage_gc_type_id());
        let dst = crate::gc_roots::shadow_stack_get(dst_slot);
        {
            let set = &mut *(dst as *mut W_SetObject);
            set.sstorage = storage as *mut u8;
            set.sstrategy = &INTEGER_SET_STRATEGY_REF;
            set.set_len_relaxed(len);
            set.hash = -1;
        }
        set_write_barrier(dst);
        return Ok(());
    }
    let keys = {
        let src = crate::gc_roots::shadow_stack_get(src_slot);
        let src_storage = int_set_storage_ptr(&*(src as *const W_SetObject));
        let mut keys = Vec::with_capacity((*src_storage).len());
        for key in (*src_storage).keys() {
            keys.push(*key);
        }
        keys
    };
    let removed = {
        let dst = crate::gc_roots::shadow_stack_get(dst_slot);
        let storage = &mut *int_set_storage_ptr(&mut *(dst as *mut W_SetObject));
        let mut removed = false;
        for key in keys {
            if storage.remove(&key).is_some() {
                removed = true;
            }
        }
        removed
    };
    if removed {
        // `delitem_with_hash` does not call `switch_to_empty_strategy`.
        publish_int_len(crate::gc_roots::shadow_stack_get(dst_slot));
    }
    Ok(())
}

/// `AbstractUnwrappedSetStrategy._difference_wrapped` when self is an int
/// set and the other strategy may contain equal elements: keep the unwrapped
/// keys `w_other.has_key` misses, still on `IntegerSetStrategy`.
///
/// # Safety
/// `dst` is on [`INTEGER_SET_STRATEGY`]. Caller holds `w_set_lock_pair`.
/// Both slots are pinned.
unsafe fn int_difference_keep_missing(
    dst_slot: usize,
    src_slot: usize,
) -> Result<(), SetUpdateError> {
    let raw = {
        let dst = crate::gc_roots::shadow_stack_get(dst_slot);
        let storage = int_set_storage_ptr(&*(dst as *const W_SetObject));
        let mut raw = Vec::with_capacity((*storage).len());
        for key in (*storage).keys() {
            raw.push(*key);
        }
        raw
    };
    let mut keep = Vec::new();
    for key in raw {
        let wrapped = object_key_for_plain_int(key);
        let _key_roots = crate::gc_roots::push_roots();
        let key_obj = crate::gc_roots::pin_root(wrapped.obj);
        let present = w_set_contains_key_for_update(
            crate::gc_roots::shadow_stack_get(src_slot),
            crate::dictmultiobject::ObjectKey {
                hash: wrapped.hash,
                obj: key_obj,
            },
        )?;
        if !present {
            keep.push(key);
        }
    }
    let mut fresh = IntSetStorage::new();
    for key in &keep {
        fresh.insert(*key, ());
    }
    let len = fresh.len();
    let storage = crate::gc_storage::gc_alloc_storage_box(fresh, int_set_storage_gc_type_id());
    let dst = crate::gc_roots::shadow_stack_get(dst_slot);
    {
        let set = &mut *(dst as *mut W_SetObject);
        set.sstorage = storage as *mut u8;
        set.sstrategy = &INTEGER_SET_STRATEGY_REF;
        set.set_len_relaxed(len);
        set.hash = -1;
    }
    set_write_barrier(dst);
    Ok(())
}

/// `AbstractUnwrappedSetStrategy._difference_update_wrapped`: walk `src`
/// and `remove` each key from `dst`. `src` may be an int set; the walk
/// goes through [`w_set_key_at`] so the key is wrapped.
///
/// # Safety
/// Caller holds `w_set_lock_pair`. Both slots are pinned.
unsafe fn difference_remove_src_keys(
    dst_slot: usize,
    src_slot: usize,
) -> Result<(), SetUpdateError> {
    let mut i = 0;
    loop {
        let src = crate::gc_roots::shadow_stack_get(src_slot);
        let Some(slot) = w_set_next_slot(src, i) else {
            break;
        };
        let src = crate::gc_roots::shadow_stack_get(src_slot);
        let Some(key) = w_set_key_at(src, slot) else {
            return Err(SetUpdateError::ChangedSize);
        };
        let src_set = &*(crate::gc_roots::shadow_stack_get(src_slot) as *const W_SetObject);
        let src_kind = src_set.sstrategy.kind;
        let src_storage = src_set.sstorage;
        let src_len = src_set.len_relaxed();
        let _key_roots = crate::gc_roots::push_roots();
        let key_obj = crate::gc_roots::pin_root(key.obj);
        w_set_remove_key_for_update(
            crate::gc_roots::shadow_stack_get(dst_slot),
            crate::dictmultiobject::ObjectKey {
                hash: key.hash,
                obj: key_obj,
            },
        )?;
        let src_set = &*(crate::gc_roots::shadow_stack_get(src_slot) as *const W_SetObject);
        if src_set.sstrategy.kind != src_kind
            || src_set.sstorage != src_storage
            || src_set.len_relaxed() != src_len
        {
            return Err(SetUpdateError::ChangedSize);
        }
        i = slot + 1;
    }
    Ok(())
}

/// `AbstractUnwrappedSetStrategy.update` when both sets are int:
/// `d_set.update(d_other)` on the `i64` tables.
///
/// # Safety
/// Both sets are on [`INTEGER_SET_STRATEGY`] and their boxes differ.
/// Caller holds `w_set_lock_pair`. `dst_slot` is pinned.
unsafe fn int_set_update_from_int(dst_slot: usize, src: PyObjectRef) -> Result<(), SetUpdateError> {
    let keys = {
        let storage = int_set_storage_ptr(&*(src as *const W_SetObject));
        let mut keys = Vec::with_capacity((*storage).len());
        for key in (*storage).keys() {
            keys.push(*key);
        }
        keys
    };
    let grew = {
        let dst = crate::gc_roots::shadow_stack_get(dst_slot);
        let storage = &mut *int_set_storage_ptr(&mut *(dst as *mut W_SetObject));
        let mut grew = false;
        for key in keys {
            if storage.insert(key, ()).is_none() {
                grew = true;
            }
        }
        grew
    };
    if grew {
        publish_int_len(crate::gc_roots::shadow_stack_get(dst_slot));
    }
    Ok(())
}

/// `ObjectSetStrategy.update` when the other set is int: iterate wrapped
/// keys into the object table. Does not switch strategy.
///
/// # Safety
/// `dst` is on [`OBJECT_SET_STRATEGY`], `src` on [`INTEGER_SET_STRATEGY`].
/// Caller holds `w_set_lock_pair`. `dst_slot` is pinned.
unsafe fn object_set_update_from_int(
    dst_slot: usize,
    src: PyObjectRef,
) -> Result<(), SetUpdateError> {
    let keys = {
        let storage = int_set_storage_ptr(&*(src as *const W_SetObject));
        let mut keys = Vec::with_capacity((*storage).len());
        for key in (*storage).keys() {
            keys.push(*key);
        }
        keys
    };
    for key in keys {
        let wrapped = object_key_for_plain_int(key);
        let _key_roots = crate::gc_roots::push_roots();
        let key_obj = crate::gc_roots::pin_root(wrapped.obj);
        w_set_insert_key_checked(
            crate::gc_roots::shadow_stack_get(dst_slot),
            crate::dictmultiobject::ObjectKey {
                hash: wrapped.hash,
                obj: key_obj,
            },
        )?;
    }
    Ok(())
}

/// # Safety
/// `dst` and `src` must point to valid `W_SetObject`s.
pub unsafe fn w_set_difference_update_from_set(
    dst: PyObjectRef,
    src: PyObjectRef,
) -> Result<(), SetUpdateError> {
    let _roots = crate::gc_roots::push_roots();
    let dst_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(dst);
    let src_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(src);
    let (_first_guard, _second_guard) = w_set_lock_pair(
        crate::gc_roots::shadow_stack_get(dst_slot),
        crate::gc_roots::shadow_stack_get(src_slot),
    );
    let dst = crate::gc_roots::shadow_stack_get(dst_slot);
    let src = crate::gc_roots::shadow_stack_get(src_slot);
    let dst_set = &*(dst as *const W_SetObject);
    let src_set = &*(src as *const W_SetObject);
    // `setobject.py EmptySetStrategy.difference_update` is a no-op.
    // `ObjectSetStrategy.may_contain_equal_elements(EmptySetStrategy)` is
    // false, so subtracting an empty operand removes nothing.
    if dst_set.sstrategy.kind == SetStrategyKind::Empty
        || src_set.sstrategy.kind == SetStrategyKind::Empty
    {
        return Ok(());
    }
    // `_difference_update_unwrapped`: the two sides sharing one storage is
    // `s -= s`, which empties self. Compare the erased word, not an
    // `ObjectSetStrategy` unerase — an int set's box is [`IntSetStorage`].
    if dst_set.sstrategy.kind == src_set.sstrategy.kind
        && !dst_set.sstorage.is_null()
        && std::ptr::eq(dst_set.sstorage, src_set.sstorage)
    {
        w_set_clear(dst);
        set_write_barrier(dst);
        return Ok(());
    }
    let dst_kind = dst_set.sstrategy.kind;
    let src_kind = src_set.sstrategy.kind;
    if dst_kind == SetStrategyKind::Int && src_kind == SetStrategyKind::Int {
        return int_difference_update(dst_slot, src_slot);
    }
    // Smaller int self builds a new int dict (`_difference_wrapped` keeps
    // this strategy). A larger int self falls through when `src` is an
    // object set and `remove` switches per non-int key.
    if dst_kind == SetStrategyKind::Int && w_set_len(dst) < w_set_len(src) {
        return int_difference_keep_missing(dst_slot, src_slot);
    }
    if src_kind == SetStrategyKind::Int && w_set_len(dst) >= w_set_len(src) {
        return difference_remove_src_keys(dst_slot, src_slot);
    }
    // setobject.py `_difference_update` — small_set -= big_set computes a fresh
    // difference by walking the smaller self storage, then replaces self's
    // storage wholesale. Besides the complexity bound, this preserves the
    // exact contains-with-hash callback direction of the upstream strategy.
    if w_set_len(dst) < w_set_len(src) {
        // The difference accumulator has no referrer until
        // `w_set_copy_storage_from` below: the `eq_w` a bucket probe runs is a
        // collection point that would otherwise sweep it, and the operand
        // bodies relocate, so each walk step reloads them.
        let result_slot = crate::gc_roots::shadow_stack_len();
        let _ = crate::gc_roots::pin_root(w_set_new());
        let dst = crate::gc_roots::shadow_stack_get(dst_slot);
        let dst_items = object_set_storage_ptr(&*(dst as *const W_SetObject));
        let dst_len = (*dst_items).len();
        let mut i = 0;
        while let Some(slot) = w_set_next_slot(crate::gc_roots::shadow_stack_get(dst_slot), i) {
            let dst = crate::gc_roots::shadow_stack_get(dst_slot);
            let Some(key) = w_set_key_at(dst, slot) else {
                return Err(SetUpdateError::ChangedSize);
            };
            if !w_set_contains_key_for_update(crate::gc_roots::shadow_stack_get(src_slot), key)? {
                let dst = crate::gc_roots::shadow_stack_get(dst_slot);
                if !same_live_object_box(&*(dst as *const W_SetObject), dst_items)
                    || (*dst_items).len() != dst_len
                {
                    return Err(SetUpdateError::ChangedSize);
                }
                // The comparison may clear or otherwise shorten `dst`.
                // The set probe restarts when `entry->key` changes;
                // once this live index disappeared there is no surviving
                // entry to copy into the difference result.
                let Some(key) = w_set_key_at(dst, slot) else {
                    return Err(SetUpdateError::ChangedSize);
                };
                w_set_insert_key_checked(crate::gc_roots::shadow_stack_get(result_slot), key)?;
            }
            i = slot + 1;
        }
        w_set_copy_storage_from(
            crate::gc_roots::shadow_stack_get(dst_slot),
            crate::gc_roots::shadow_stack_get(result_slot),
        );
        return Ok(());
    }
    // `src` is a distinct storage, so removing from `dst` cannot renumber it.
    let src = crate::gc_roots::shadow_stack_get(src_slot);
    let src_items = object_set_storage_ptr(&*(src as *const W_SetObject));
    let src_len = (*src_items).len();
    let mut i = 0;
    while let Some(slot) = w_set_next_slot(crate::gc_roots::shadow_stack_get(src_slot), i) {
        let src = crate::gc_roots::shadow_stack_get(src_slot);
        let Some(key) = w_set_key_at(src, slot) else {
            return Err(SetUpdateError::ChangedSize);
        };
        w_set_remove_key_for_update(crate::gc_roots::shadow_stack_get(dst_slot), key)?;
        let src = crate::gc_roots::shadow_stack_get(src_slot);
        if !same_live_object_box(&*(src as *const W_SetObject), src_items)
            || (*src_items).len() != src_len
        {
            return Err(SetUpdateError::ChangedSize);
        }
        i = slot + 1;
    }
    Ok(())
}

/// Merge a set operand's storage in, keeping the digests it holds.
///
/// `setobject.py ObjectSetStrategy.update` takes `d_obj.update(
/// d_other)` when the operand shares this strategy — labelled "optimization
/// only" upstream, but it is also what keeps a set operand's elements from
/// being handed to a user `__hash__` a second time. Elements equal across the
/// two sides still meet in a bucket, so `eq_w` runs and can raise.
///
/// # Safety
/// `dst` and `src` must point to valid `W_SetObject`s.
pub unsafe fn w_set_update_from_set(
    dst: PyObjectRef,
    src: PyObjectRef,
) -> Result<(), SetUpdateError> {
    // The bodies relocate: `w_set_insert_key_into` runs `eq_w`, and this
    // frame's `dst`/`src` locals are the only referrers for a temporary.
    let _roots = crate::gc_roots::push_roots();
    let dst_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(dst);
    let src_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(src);
    let (_first_guard, _second_guard) = w_set_lock_pair(
        crate::gc_roots::shadow_stack_get(dst_slot),
        crate::gc_roots::shadow_stack_get(src_slot),
    );
    let dst = crate::gc_roots::shadow_stack_get(dst_slot);
    let src = crate::gc_roots::shadow_stack_get(src_slot);
    let dst_kind = (*(dst as *const W_SetObject)).sstrategy.kind;
    let src_kind = (*(src as *const W_SetObject)).sstrategy.kind;
    // `setobject.py EmptySetStrategy.update` steals the other's strategy
    // and `get_storage_copy`. An empty operand contributes nothing
    // (`ObjectSetStrategy.update` iterates it and stops).
    if dst_kind == SetStrategyKind::Empty {
        if src_kind != SetStrategyKind::Empty {
            w_set_copy_storage_from(dst, src);
        }
        return Ok(());
    }
    if src_kind == SetStrategyKind::Empty {
        return Ok(());
    }
    // Same erased box: nothing to merge. Do not unerase an int box as
    // [`SetItemsStorage`].
    if dst_kind == src_kind
        && !(*(dst as *const W_SetObject)).sstorage.is_null()
        && std::ptr::eq(
            (*(dst as *const W_SetObject)).sstorage,
            (*(src as *const W_SetObject)).sstorage,
        )
    {
        return Ok(());
    }
    if dst_kind == SetStrategyKind::Int && src_kind == SetStrategyKind::Int {
        return int_set_update_from_int(dst_slot, src);
    }
    // `AbstractUnwrappedSetStrategy.update`: a different strategy switches
    // to object and retries. `ObjectSetStrategy.update` does not switch; it
    // inserts the other side's wrapped keys.
    if dst_kind == SetStrategyKind::Int {
        switch_int_to_object_strategy(crate::gc_roots::shadow_stack_get(dst_slot));
    }
    let dst = crate::gc_roots::shadow_stack_get(dst_slot);
    let src = crate::gc_roots::shadow_stack_get(src_slot);
    if (*(src as *const W_SetObject)).sstrategy.kind == SetStrategyKind::Int {
        return object_set_update_from_int(dst_slot, src);
    }
    // Both tables are captured once for the whole merge — `update` unerases
    // `d_obj` up front (`setobject.py`) and `d_obj.update(d_other)` runs
    // `ll_dict_update(dic1, dic2)` on those two tables (`rordereddict.py`).
    // A callback that clears either set swaps its live storage; the merge keeps
    // reading the captured source and inserting into the captured destination,
    // both now orphaned snapshots.
    //
    // `src`'s keys are still read one index at a time rather than collected: an
    // `eq_w` raised from the bucket probe below can move every element, and
    // the collector rewrites the `obj` slots inside the two tables in place
    // (`set_items_storage_custom_trace`) — a `Vec` of keys lifted out of them
    // would not be walked and would be left holding stale pointers.
    let dst_items = capture_set_items(dst);
    let src_items = capture_set_items(src);
    let mut i = 0;
    while let Some((slot, &key, _)) = (*src_items).next_entry(i) {
        w_set_insert_key_into(crate::gc_roots::shadow_stack_get(dst_slot), dst_items, key)?;
        i = slot + 1;
    }
    Ok(())
}

/// Failure modes of the PyPy `ObjectSetStrategy.update` table merge.
pub enum SetUpdateError {
    /// An equality callback raised; its concrete exception is parked in the
    /// interpreter's pending dict-key error slot.
    Key(crate::dictmultiobject::DictKeyError),
    /// A callback changed one of the tables while it was being traversed.
    ChangedSize,
}

/// Insert one cached-hash key without holding a table borrow across user
/// `eq_w`.
///
/// The set's storage box is captured and pinned once at entry
/// (`capture_set_items`), mirroring PyPy's `d = self.unerase(w_set.sstorage)`
/// (`AbstractUnwrappedSetStrategy.add`): the membership probe and the
/// follow-up insert both run against that box.  If a probing `__eq__` clears
/// the set, `switch_to_empty_strategy` nulls the live word while this insert
/// still targets the orphaned snapshot, so the element lands in the dropped
/// box and is lost — the behaviour PyPy exhibits when `add` captures its
/// storage before a re-entrant `clear`.  The scan drops the table borrow before each
/// comparison and inserts through an already-proven vacant raw entry, so
/// insertion performs no second user callback.
unsafe fn w_set_insert_key_reentrant(
    dst: PyObjectRef,
    key: crate::dictmultiobject::ObjectKey,
) -> Result<(), SetUpdateError> {
    let _roots = crate::gc_roots::push_roots();
    let items = capture_set_items(dst);
    w_set_insert_key_into(dst, items, key)
}

/// Probe-and-insert half of [`w_set_insert_key_reentrant`] against a storage
/// box the caller already captured and pinned.  `w_set_update_from_set` passes
/// the box it captured for the whole merge so every source key targets the same
/// (possibly orphaned) table, the way `ll_dict_update` keeps inserting into its
/// captured `dic1`.
unsafe fn w_set_insert_key_into(
    dst: PyObjectRef,
    items: *mut SetItemsStorage,
    key: crate::dictmultiobject::ObjectKey,
) -> Result<(), SetUpdateError> {
    let _dst_roots = crate::gc_roots::push_roots();
    let dst_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(dst);
    let key_slot = crate::gc_roots::shadow_stack_len();
    let mut key = key;
    key.obj = crate::gc_roots::pin_root(key.obj);
    // Single insert probe (matches `r_dict.setitem`'s one bucket scan), run
    // callback-free so no user `__eq__` mutates the set while the table
    // borrow is live.  When every same-hash comparison stays inside the
    // builtin ladder the probe appends in place; a pair it cannot decide
    // breaks the probe, withholds the store, and re-runs the operation over
    // `scan_set_key_reentrant` below.  Without this the membership half of
    // every `add` walks the table entry by entry, which is quadratic in the
    // set's size.
    if let Some(result) = callback_free_set_op(|| {
        let entries = &mut *items;
        let index = entries.index_of(&key);
        if crate::dict_eq_hook::callback_free_probe_broken() {
            return false;
        }
        if index.is_some() {
            return false;
        }
        // The probe above proved no bucket entry compares equal without
        // leaving the ladder, so this placement probe repeats those same
        // comparisons and cannot break either.
        let mut live_key = key;
        live_key.obj = crate::gc_roots::shadow_stack_get(key_slot);
        entries.insert(live_key, ());
        true
    }) {
        if result.map_err(SetUpdateError::Key)? {
            // The key landed in `items`, which a probing `clear` may already
            // have detached from `dst`. Barrier that box, not the set body.
            // The live length stays the 0 `clear` published.
            set_items_write_barrier(items);
            let dst = crate::gc_roots::shadow_stack_get(dst_slot);
            publish_len_if_live_box(dst, items);
        }
        return Ok(());
    }

    key.obj = crate::gc_roots::shadow_stack_get(key_slot);
    let (found, key) = scan_set_key_reentrant(items, key).map_err(SetUpdateError::Key)?;
    if found.is_some() {
        return Ok(());
    }
    // The scan above proved the key absent, and repeating the probe here would
    // repeat its comparisons — the ones that can re-enter this table.  Place it
    // on the digest alone (`ll_call_insert_clean_function`).
    (*items).insert_known_absent(key, ());
    // Same box the scan captured. A `clear` inside that scan has already
    // nulled `dst.sstorage`; the new key lives here, and the live length
    // stays the 0 `clear` published.
    set_items_write_barrier(items);
    let dst = crate::gc_roots::shadow_stack_get(dst_slot);
    publish_len_if_live_box(dst, items);
    Ok(())
}

/// Membership half of `contains_with_hash` for a mutation-sensitive set
/// operation.  As with `rordereddict.ll_dict_lookup`, no table borrow crosses
/// `eq_w`, and a callback that replaces an entry restarts the lookup.
unsafe fn w_set_contains_key_for_update(
    probe: PyObjectRef,
    mut key: crate::dictmultiobject::ObjectKey,
) -> Result<bool, SetUpdateError> {
    let _probe_roots = crate::gc_roots::push_roots();
    let probe_slot = crate::gc_roots::shadow_stack_len();
    let probe = crate::gc_roots::pin_root(probe);
    let key_root = crate::gc_roots::shadow_stack_len();
    key.obj = crate::gc_roots::pin_root(key.obj);
    // `EmptySetStrategy.has_key` is False once the key is hashed. The digest
    // is already on `key`.
    if (*(probe as *const W_SetObject)).sstrategy.kind == SetStrategyKind::Empty {
        return Ok(false);
    }
    // `AbstractUnwrappedSetStrategy.has_key`: a plain int probes the `i64`
    // table; any other key switches to object and retries.
    if (*(probe as *const W_SetObject)).sstrategy.kind == SetStrategyKind::Int {
        if crate::listobject::is_plain_int1(key.obj) {
            return Ok(int_set_contains_unwrapped(probe, key));
        }
        switch_int_to_object_strategy(crate::gc_roots::shadow_stack_get(probe_slot));
    }
    let probe = crate::gc_roots::shadow_stack_get(probe_slot);
    key.obj = crate::gc_roots::shadow_stack_get(key_root);
    // Bucket probe first, as in `w_set_contains_key_checked`.  The walk below
    // is the reentrant fallback and visits every entry, so without this a
    // whole-set difference probes linearly per element and runs quadratic.
    if let Some(result) = callback_free_set_op(|| {
        let s = &*(probe as *const W_SetObject);
        if s.sstrategy.kind == SetStrategyKind::Empty {
            return false;
        }
        (*object_set_storage_ptr(s)).contains_key(&key)
    }) {
        return result.map_err(SetUpdateError::Key);
    }
    'restart: loop {
        let probe = crate::gc_roots::shadow_stack_get(probe_slot);
        key.obj = crate::gc_roots::shadow_stack_get(key_root);
        if (*(probe as *const W_SetObject)).sstrategy.kind == SetStrategyKind::Empty {
            return Ok(false);
        }
        let items = object_set_storage_ptr(&*(probe as *const W_SetObject));
        let len = (*items).len();
        let mut i = 0;
        loop {
            let Some((slot, &stored, _)) = (*items).next_entry(i) else {
                break;
            };
            if stored.hash == key.hash {
                let _roots = crate::gc_roots::push_roots();
                let stored_slot = crate::gc_roots::shadow_stack_len();
                let stored_obj = crate::gc_roots::pin_root(stored.obj);
                let key_slot = crate::gc_roots::shadow_stack_len();
                key.obj = crate::gc_roots::pin_root(key.obj);
                let equal = crate::dictmultiobject::dict_keys_equal(stored_obj, key.obj);
                let stored_obj = crate::gc_roots::shadow_stack_get(stored_slot);
                key.obj = crate::gc_roots::shadow_stack_get(key_slot);
                if crate::dictmultiobject::take_dict_key_error() {
                    return Err(SetUpdateError::Key(crate::dictmultiobject::DictKeyError));
                }
                let probe = crate::gc_roots::shadow_stack_get(probe_slot);
                if !same_live_object_box(&*(probe as *const W_SetObject), items)
                    || (*items).len() != len
                    || !(*items).get_slot(slot).is_some_and(|(current, _)| {
                        current.hash == stored.hash && current.obj == stored_obj
                    })
                {
                    continue 'restart;
                }
                if equal {
                    return Ok(true);
                }
            }
            i = slot + 1;
        }
        return Ok(false);
    }
}

/// `delitem_with_hash` for a mutation-sensitive difference update.  Find the
/// matching bucket without lending the table across Python code, restart after
/// a hostile comparison like `ll_dict_lookup`, then delete the proven slot
/// without another equality callback.
unsafe fn w_set_remove_key_for_update(
    dst: PyObjectRef,
    mut key: crate::dictmultiobject::ObjectKey,
) -> Result<(), SetUpdateError> {
    let _dst_roots = crate::gc_roots::push_roots();
    let dst_slot = crate::gc_roots::shadow_stack_len();
    let dst = crate::gc_roots::pin_root(dst);
    let key_root = crate::gc_roots::shadow_stack_len();
    key.obj = crate::gc_roots::pin_root(key.obj);
    // An empty set has nothing to delete. `EmptySetStrategy.remove` is the
    // public discard path; this helper is `delitem_with_hash` and does not
    // switch strategy when the last element leaves.
    if (*(dst as *const W_SetObject)).sstrategy.kind == SetStrategyKind::Empty {
        return Ok(());
    }
    // `AbstractUnwrappedSetStrategy.remove` / `delitem_with_hash`. A plain
    // int is deleted from the `i64` table and the strategy stays put even
    // when the dict becomes empty. Any other key switches, then the object
    // path below deletes it.
    if (*(dst as *const W_SetObject)).sstrategy.kind == SetStrategyKind::Int {
        if crate::listobject::is_plain_int1(key.obj) {
            int_set_remove_unwrapped(dst, key, false);
            return Ok(());
        }
        switch_int_to_object_strategy(crate::gc_roots::shadow_stack_get(dst_slot));
    }
    let dst = crate::gc_roots::shadow_stack_get(dst_slot);
    key.obj = crate::gc_roots::shadow_stack_get(key_root);
    // Locate the bucket callback-free before falling back to the entry walk,
    // which is linear in the set's size.  The index is resolved inside the
    // probe and the removal withheld when a comparison leaves the builtin
    // ladder, so the walk below can redo the whole operation.
    if let Some(result) = callback_free_set_op(|| {
        let set = &*(dst as *const W_SetObject);
        if set.sstrategy.kind == SetStrategyKind::Empty {
            return false;
        }
        let items = object_set_storage_ptr(set);
        let index = (*items).index_of(&key);
        if crate::dict_eq_hook::callback_free_probe_broken() {
            return false;
        }
        if let Some(index) = index {
            set_remove_slot(items, index);
            true
        } else {
            false
        }
    }) {
        if result.map_err(SetUpdateError::Key)? {
            let dst = crate::gc_roots::shadow_stack_get(dst_slot);
            let set = &mut *(dst as *mut W_SetObject);
            set.set_len_relaxed(set.len_relaxed() - 1);
            set.hash = -1;
        }
        return Ok(());
    }
    'restart: loop {
        let dst = crate::gc_roots::shadow_stack_get(dst_slot);
        key.obj = crate::gc_roots::shadow_stack_get(key_root);
        if (*(dst as *const W_SetObject)).sstrategy.kind == SetStrategyKind::Empty {
            return Ok(());
        }
        let items = object_set_storage_ptr(&*(dst as *const W_SetObject));
        let len = (*items).len();
        let mut found = None;
        let mut i = 0;
        loop {
            let Some((slot, &stored, _)) = (*items).next_entry(i) else {
                break;
            };
            if stored.hash == key.hash {
                let _roots = crate::gc_roots::push_roots();
                let stored_slot = crate::gc_roots::shadow_stack_len();
                let stored_obj = crate::gc_roots::pin_root(stored.obj);
                let key_slot = crate::gc_roots::shadow_stack_len();
                key.obj = crate::gc_roots::pin_root(key.obj);
                let equal = crate::dictmultiobject::dict_keys_equal(stored_obj, key.obj);
                let stored_obj = crate::gc_roots::shadow_stack_get(stored_slot);
                key.obj = crate::gc_roots::shadow_stack_get(key_slot);
                if crate::dictmultiobject::take_dict_key_error() {
                    return Err(SetUpdateError::Key(crate::dictmultiobject::DictKeyError));
                }
                let dst = crate::gc_roots::shadow_stack_get(dst_slot);
                if !same_live_object_box(&*(dst as *const W_SetObject), items)
                    || (*items).len() != len
                    || !(*items).get_slot(slot).is_some_and(|(current, _)| {
                        current.hash == stored.hash && current.obj == stored_obj
                    })
                {
                    continue 'restart;
                }
                if equal {
                    found = Some(slot);
                    break;
                }
            }
            i = slot + 1;
        }
        if let Some(index) = found {
            set_remove_slot(items, index);
            let dst = crate::gc_roots::shadow_stack_get(dst_slot);
            let set = &mut *(dst as *mut W_SetObject);
            set.set_len_relaxed(set.len_relaxed() - 1);
            set.hash = -1;
        }
        return Ok(());
    }
}

/// Number of elements in the set.
///
/// Takes no stripe lock, matching `Objects/setobject.c set_len`:
/// `FT_ATOMIC_LOAD_SSIZE_RELAXED(so->used)`.  A lock here would be stricter
/// than the spec and would not buy anything — every mutator publishes the
/// count with a relaxed store while holding the stripe, so the value read is
/// from one side of a concurrent mutation either way, and the JIT's `len` fold
/// lowers to this same unlocked load.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_len(obj: PyObjectRef) -> usize {
    (*(obj as *const W_SetObject)).len_relaxed()
}

/// Cached frozenset hash; `-1` is the uncomputed sentinel.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_frozenset_cached_hash(obj: PyObjectRef) -> Option<i64> {
    let _set_guard = w_set_lock(obj);
    let hash = (*(obj as *const W_SetObject)).hash;
    (hash != -1).then_some(hash)
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_frozenset_set_cached_hash(obj: PyObjectRef, hash: i64) {
    let _set_guard = w_set_lock(obj);
    (*(obj as *mut W_SetObject)).hash = hash;
}

/// Digests already carried by the r_dict keys. Python 3.14 frozenset hashing
/// consumes these instead of invoking each element's `__hash__` again.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_set_stored_hashes(obj: PyObjectRef) -> Vec<i64> {
    let _set_guard = w_set_lock(obj);
    // `EmptySetStrategy.iter` yields nothing, so an empty frozenset hashes
    // as the empty fold. `IntegerSetStrategy` stores `i64` keys; the fold
    // wants the digest `hash_w` would store (`intobject.py _hash_int`),
    // via [`object_key_for`]. Copy the keys out before that allocation so
    // the set borrow does not cross it.
    let keys = {
        let s = &*(obj as *const W_SetObject);
        match s.sstrategy.kind {
            SetStrategyKind::Empty => return Vec::new(),
            SetStrategyKind::Int => {
                let mut keys = Vec::with_capacity((*int_set_storage_ptr(s)).len());
                for key in (*int_set_storage_ptr(s)).keys() {
                    keys.push(*key);
                }
                Some(keys)
            }
            SetStrategyKind::Object => {
                return (*object_set_storage_ptr(s))
                    .keys()
                    .map(|key| key.hash)
                    .collect();
            }
            SetStrategyKind::Bytes | SetStrategyKind::Ascii | SetStrategyKind::Identity => {
                return Vec::new();
            }
        }
    };
    let mut hashes = Vec::with_capacity(keys.as_ref().map(|k| k.len()).unwrap_or(0));
    for key in keys.unwrap_or_default() {
        hashes.push(object_key_for_plain_int(key).hash);
    }
    hashes
}

/// The key in `slot`, carrying the digest it was stored under, or `None` when
/// nothing occupies it — past the last entry, or in the hole a delete left.
/// [`w_set_next_slot`] finds the slots that answer.
///
/// `setobject.py iterkeys_with_hash` walks a storage handing out
/// `(key, keyhash)` pairs so the walk's consumer can place or probe the key
/// without hashing it again. Reading one index at a time lets a caller whose
/// loop body reaches user code (an `eq_w` from a bucket probe) re-read the key
/// afterwards: the collector rewrites the `obj` slots inside the table in place
/// (`set_items_storage_custom_trace`), so a key read before that point can be
/// stale while the table itself stays correct.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_key_at(
    obj: PyObjectRef,
    slot: usize,
) -> Option<crate::dictmultiobject::ObjectKey> {
    let _set_guard = w_set_lock(obj);
    // `IntegerIteratorImplementation.next_entry` is `space.newint`. The
    // digest is `hash_w` (`intobject.py _hash_int`: `hash(1) == 1`,
    // `hash(-1) == -2`), via [`object_key_for_plain_int`]. The `i64` is
    // copied out before that allocation.
    let raw = {
        let s = &*(obj as *const W_SetObject);
        match s.sstrategy.kind {
            SetStrategyKind::Empty => return None,
            SetStrategyKind::Int => (*int_set_storage_ptr(s))
                .get_slot(slot)
                .map(|(key, _)| *key),
            SetStrategyKind::Object => {
                return (*object_set_storage_ptr(s))
                    .get_slot(slot)
                    .map(|(&key, _)| key);
            }
            SetStrategyKind::Bytes | SetStrategyKind::Ascii | SetStrategyKind::Identity => {
                return None;
            }
        }
    };
    match raw {
        Some(key) => Some(object_key_for_plain_int(key)),
        None => None,
    }
}

/// `num_ever_used_items` (`rordereddict.py` `_ll_dictnext`): one past the
/// highest slot a walk may name. Dead slots below it read as null from
/// [`w_set_iterkey_at`].
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`) so the
/// entries-length read stays inside this residual and the caller's merge
/// loop only sees a word.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
#[majit_macros::dont_look_inside]
pub unsafe fn w_set_num_ever_used_items(obj: *mut PyObject) -> usize {
    let _set_guard = w_set_lock(obj);
    let s = &*(obj as *const W_SetObject);
    match s.sstrategy.kind {
        SetStrategyKind::Empty => 0,
        SetStrategyKind::Int => (*int_set_storage_ptr(s)).entry_slots(),
        SetStrategyKind::Object => (*object_set_storage_ptr(s)).entry_slots(),
        SetStrategyKind::Bytes | SetStrategyKind::Ascii | SetStrategyKind::Identity => 0,
    }
}

/// One step of `iterkeys_with_hash` (`rlib/objectmodel.py`): the key stored
/// at `index`, or null when that slot is dead or past the end.
///
/// The slot read (`RDict::get_slot`, the `get_index` lookup the merge used to
/// inline) is residual (`@jit.dont_look_inside`, `rlib/jit.py`) so a traced
/// merge loop does not lower it. The digest half is [`w_set_iterkey_hash_at`].
/// Parameters are `*mut PyObject` so the word-ABI trampoline is emitted; a
/// `PyObjectRef` alias is not that token.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
#[majit_macros::dont_look_inside]
pub unsafe fn w_set_iterkey_at(obj: *mut PyObject, index: usize) -> *mut PyObject {
    let _set_guard = w_set_lock(obj);
    let raw = {
        let s = &*(obj as *const W_SetObject);
        match s.sstrategy.kind {
            SetStrategyKind::Empty => return std::ptr::null_mut(),
            SetStrategyKind::Int => (*int_set_storage_ptr(s))
                .get_slot(index)
                .map(|(key, _)| *key),
            SetStrategyKind::Object => {
                return match (*object_set_storage_ptr(s)).get_slot(index) {
                    Some((key, _)) => key.obj,
                    None => std::ptr::null_mut(),
                };
            }
            SetStrategyKind::Bytes | SetStrategyKind::Ascii | SetStrategyKind::Identity => {
                return std::ptr::null_mut();
            }
        }
    };
    match raw {
        Some(key) => object_key_for_plain_int(key).obj,
        None => std::ptr::null_mut(),
    }
}

/// Digest half of [`w_set_iterkey_at`]. Only meaningful when that call
/// returned a key. Same residual boundary.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`, and `index` must name a live slot.
#[majit_macros::dont_look_inside]
pub unsafe fn w_set_iterkey_hash_at(obj: *mut PyObject, index: usize) -> i64 {
    let _set_guard = w_set_lock(obj);
    let raw = {
        let s = &*(obj as *const W_SetObject);
        match s.sstrategy.kind {
            SetStrategyKind::Empty => return 0,
            SetStrategyKind::Int => (*int_set_storage_ptr(s))
                .get_slot(index)
                .map(|(key, _)| *key),
            SetStrategyKind::Object => {
                return match (*object_set_storage_ptr(s)).get_slot(index) {
                    Some((key, _)) => key.hash,
                    None => 0,
                };
            }
            SetStrategyKind::Bytes | SetStrategyKind::Ascii | SetStrategyKind::Identity => {
                return 0;
            }
        }
    };
    match raw {
        Some(key) => object_key_for_plain_int(key).hash,
        None => 0,
    }
}

/// `contains_with_hash` (`rlib/objectmodel.py` / `ll_dict_contains_with_hash`).
///
/// Returns `1` when present, `0` when absent, and `-1` when user `__eq__`
/// raised. The concrete exception stays in the pending dict-key slot, the
/// same channel [`w_set_contains_key_checked`] uses. The reentrant scan lives
/// inside this residual so the caller's loop does not lower it.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`, and `hash` must be the digest
/// `key` was stored under.
#[majit_macros::dont_look_inside]
pub unsafe fn w_set_contains_with_hash(obj: *mut PyObject, key: *mut PyObject, hash: i64) -> i64 {
    match w_set_contains_key_checked(obj, crate::dictmultiobject::ObjectKey { hash, obj: key }) {
        Ok(true) => 1,
        Ok(false) => 0,
        Err(_) => -1,
    }
}

/// `setitem_with_hash` (`rlib/objectmodel.py` / `ll_dict_setitem_with_hash`)
/// placing `None` as the set value.
///
/// Returns `0` on success, `-1` when user `__eq__` raised (the exception is
/// in the pending dict-key slot), and `-2` when a callback resized the table
/// (`SetUpdateError::ChangedSize`). Absence was
/// already decided by [`w_set_contains_with_hash`]; insertion still probes,
/// because a callback may have changed the table, and that probe is
/// [`w_set_insert_key_checked`] (callback-free, then
/// [`scan_set_key_reentrant`]).
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`, and `hash` must be the digest
/// `key` was stored under.
#[majit_macros::dont_look_inside]
pub unsafe fn w_set_setitem_with_hash(obj: *mut PyObject, key: *mut PyObject, hash: i64) -> i64 {
    match w_set_insert_key_checked(obj, crate::dictmultiobject::ObjectKey { hash, obj: key }) {
        Ok(()) => 0,
        Err(SetUpdateError::Key(_)) => -1,
        Err(SetUpdateError::ChangedSize) => -2,
    }
}

/// Snapshot the contained elements as a `Vec`.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_items(obj: PyObjectRef) -> Vec<PyObjectRef> {
    let _set_guard = w_set_lock(obj);
    // `setobject.py EmptySetStrategy.getkeys` is `[]`.
    // `IntegerSetStrategy.getkeys` wraps each unwrapped key. Copy the
    // `i64`s out before `w_int_new`.
    let keys = {
        let s = &*(obj as *const W_SetObject);
        match s.sstrategy.kind {
            SetStrategyKind::Empty => return Vec::new(),
            SetStrategyKind::Int => {
                let mut keys = Vec::with_capacity((*int_set_storage_ptr(s)).len());
                for key in (*int_set_storage_ptr(s)).keys() {
                    keys.push(*key);
                }
                Some(keys)
            }
            SetStrategyKind::Object => {
                let mut items = Vec::with_capacity((*object_set_storage_ptr(s)).len());
                for key in (*object_set_storage_ptr(s)).keys() {
                    items.push(key.obj);
                }
                return items;
            }
            SetStrategyKind::Bytes | SetStrategyKind::Ascii | SetStrategyKind::Identity => {
                return Vec::new();
            }
        }
    };
    let roots = crate::gc_roots::push_roots();
    let base = roots.base();
    let mut len = 0usize;
    for key in keys.unwrap_or_default() {
        let wrapped = object_key_for_plain_int(key);
        let _ = roots.pin_root(wrapped.obj);
        len += 1;
    }
    (0..len).map(|i| roots.get(base + i)).collect()
}

/// Walk, in place, every element `PyObjectRef` slot of a set for an
/// immortal owner.  Forwards each `ObjectKey.obj` slot: `ObjectKey.hash` is
/// identity-stable across a GC move, so writing the relocated pointer through
/// the key's `obj` slot keeps the bucket index valid.  Alloc-free — unlike
/// [`w_set_items`], which materialises a `Vec`.  A collector-owned set does
/// not use this walk: `set_items_storage_custom_trace` traces the storage
/// box, and the set body only forwards `items`.
///
/// # Safety
/// `obj` must point to a valid `W_SetObject`.
pub unsafe fn w_set_walk_gc_refs(obj: PyObjectRef, visitor: &mut dyn FnMut(*mut PyObjectRef)) {
    let set = &mut *(obj as *mut W_SetObject);
    // `EmptySetStrategy.get_empty_storage` is null. Do not unerase it.
    // `IntegerSetStrategy` keys are `i64`: there is no `PyObjectRef` to
    // visit (`IntDictStrategy.walk_gc_refs` skips the key half; a set has
    // no value half either). Kind and storage are read together; this walk
    // runs from an immortal owner the way `w_dict_walk_gc_refs` does,
    // without the stripe.
    if set.sstrategy.kind != SetStrategyKind::Object || set.sstorage.is_null() {
        return;
    }
    let entries = &mut *object_set_storage_ptr(set);
    for (key, _) in entries.iter_mut_for_trace() {
        let key_ptr = key as *const crate::dictmultiobject::ObjectKey
            as *mut crate::dictmultiobject::ObjectKey;
        visitor(std::ptr::addr_of_mut!((*key_ptr).obj) as *mut PyObjectRef);
    }
    visitor(entries.entries_slot() as *mut PyObjectRef);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intobject::w_int_new;

    fn install_test_hash_hook() {
        unsafe fn hash_int(obj: PyObjectRef) -> i64 {
            // Bool is not a plain int (`is_plain_int1`). A str must not be
            // read as `W_IntObject`. Exact ints hash to their value here;
            // production `hash_w` is `intobject.py _hash_int`.
            if crate::is_bool(obj) {
                return crate::w_bool_get_value(obj) as i64;
            }
            if crate::py_type_check(obj, &crate::INT_TYPE) {
                return crate::w_int_get_value(obj);
            }
            0
        }

        unsafe fn hash_str(_ptr: *const u8, _len: usize) -> i64 {
            0
        }

        crate::dict_eq_hook::register_hash_w_hook(hash_int);
        crate::dict_eq_hook::register_hash_str_hook(hash_str);
    }

    #[test]
    fn add_dedupes() {
        install_test_hash_hook();
        let s = w_set_new();
        unsafe {
            assert_eq!(
                (*(s as *const W_SetObject)).sstrategy.kind,
                SetStrategyKind::Empty
            );
            w_set_add(s, w_int_new(1));
            w_set_add(s, w_int_new(1));
            w_set_add(s, w_int_new(2));
            assert_eq!(
                (*(s as *const W_SetObject)).sstrategy.kind,
                SetStrategyKind::Int
            );
            assert_eq!(w_set_len(s), 2);
            assert!(w_set_contains(s, w_int_new(1)));
            assert!(w_set_contains(s, w_int_new(2)));
            assert!(!w_set_contains(s, w_int_new(3)));
        }
    }

    #[test]
    fn discard_removes() {
        install_test_hash_hook();
        let s = w_set_new();
        unsafe {
            w_set_add(s, w_int_new(1));
            w_set_add(s, w_int_new(2));
            assert!(w_set_discard(s, w_int_new(1)));
            assert!(!w_set_discard(s, w_int_new(99)));
            assert_eq!(w_set_len(s), 1);
            assert!(w_set_contains(s, w_int_new(2)));
        }
    }

    #[test]
    fn discard_from_the_front_keeps_later_members() {
        install_test_hash_hook();
        let s = w_set_new();
        unsafe {
            for i in 0..64 {
                w_set_add(s, w_int_new(i));
            }
            for i in 0..32 {
                assert!(w_set_discard(s, w_int_new(i)));
            }
            assert_eq!(w_set_len(s), 32);
            for i in 0..32 {
                assert!(!w_set_contains(s, w_int_new(i)));
            }
            for i in 32..64 {
                assert!(w_set_contains(s, w_int_new(i)));
            }
        }
    }

    #[test]
    fn frozenset_distinct_type() {
        let s = w_set_new();
        let fs = w_frozenset_new();
        unsafe {
            assert!(is_set(s));
            assert!(!is_frozenset(s));
            assert!(is_frozenset(fs));
            assert!(!is_set(fs));
        }
    }

    #[test]
    fn fresh_set_is_empty_with_null_storage() {
        let s = w_set_new();
        let fs = w_frozenset_new();
        unsafe {
            let set = &*(s as *const W_SetObject);
            assert_eq!(set.sstrategy.kind, SetStrategyKind::Empty);
            assert!(set.sstorage.is_null());
            assert_eq!(w_set_len(s), 0);
            let frozen = &*(fs as *const W_SetObject);
            assert_eq!(frozen.sstrategy.kind, SetStrategyKind::Empty);
            assert!(frozen.sstorage.is_null());
            assert_eq!(w_set_len(fs), 0);
        }
    }

    #[test]
    fn first_plain_int_add_promotes_to_int() {
        install_test_hash_hook();
        let s = w_set_new();
        unsafe {
            w_set_add(s, w_int_new(1));
            let set = &*(s as *const W_SetObject);
            assert_eq!(set.sstrategy.kind, SetStrategyKind::Int);
            assert!(!set.sstorage.is_null());
            assert_eq!(w_set_len(s), 1);
            assert!(w_set_contains(s, w_int_new(1)));
        }
    }

    #[test]
    fn clear_returns_to_empty() {
        install_test_hash_hook();
        let s = w_set_new();
        unsafe {
            w_set_add(s, w_int_new(1));
            w_set_add(s, w_int_new(2));
            w_set_clear(s);
            let set = &*(s as *const W_SetObject);
            assert_eq!(set.sstrategy.kind, SetStrategyKind::Empty);
            assert!(set.sstorage.is_null());
            assert_eq!(w_set_len(s), 0);
            assert!(!w_set_contains(s, w_int_new(1)));
            // `EmptySetStrategy.clear` is a no-op.
            w_set_clear(s);
            assert_eq!(
                (*(s as *const W_SetObject)).sstrategy.kind,
                SetStrategyKind::Empty
            );
            assert!((*(s as *const W_SetObject)).sstorage.is_null());
        }
    }

    #[test]
    fn discard_of_last_element_returns_to_empty() {
        install_test_hash_hook();
        let s = w_set_new();
        unsafe {
            w_set_add(s, w_int_new(1));
            w_set_add(s, w_int_new(2));
            assert!(w_set_discard(s, w_int_new(1)));
            assert_eq!(
                (*(s as *const W_SetObject)).sstrategy.kind,
                SetStrategyKind::Int
            );
            assert!(w_set_discard(s, w_int_new(2)));
            let set = &*(s as *const W_SetObject);
            assert_eq!(set.sstrategy.kind, SetStrategyKind::Empty);
            assert!(set.sstorage.is_null());
            assert_eq!(w_set_len(s), 0);
        }
    }

    #[test]
    fn contains_and_discard_on_empty_are_false() {
        install_test_hash_hook();
        let s = w_set_new();
        unsafe {
            assert!(!w_set_contains(s, w_int_new(1)));
            assert!(!w_set_discard(s, w_int_new(1)));
            let set = &*(s as *const W_SetObject);
            assert_eq!(set.sstrategy.kind, SetStrategyKind::Empty);
            assert!(set.sstorage.is_null());
            assert!(w_set_popitem(s).is_none());
            assert!(w_set_next_slot(s, 0).is_none());
            assert!(w_set_key_at(s, 0).is_none());
            assert!(w_set_items(s).is_empty());
            assert!(w_set_stored_hashes(s).is_empty());
            assert_eq!(w_set_num_ever_used_items(s), 0);
            assert!(w_set_iterkey_at(s, 0).is_null());
            assert_eq!(w_set_iterkey_hash_at(s, 0), 0);
        }
    }

    #[test]
    fn empty_update_copy_and_difference_do_not_deref_null() {
        install_test_hash_hook();
        unsafe {
            let empty = w_set_new();
            let other = w_set_new();
            w_set_add(other, w_int_new(7));
            assert!(w_set_update_from_set(empty, other).is_ok());
            assert_eq!(
                (*(empty as *const W_SetObject)).sstrategy.kind,
                SetStrategyKind::Int
            );
            assert_eq!(w_set_len(empty), 1);
            assert!(w_set_contains(empty, w_int_new(7)));

            let empty2 = w_set_new();
            assert!(w_set_update_from_set(other, empty2).is_ok());
            assert_eq!(w_set_len(other), 1);
            assert_eq!(
                (*(other as *const W_SetObject)).sstrategy.kind,
                SetStrategyKind::Int
            );

            let dst = w_set_new();
            w_set_add(dst, w_int_new(1));
            w_set_copy_storage_from(dst, empty2);
            assert_eq!(
                (*(dst as *const W_SetObject)).sstrategy.kind,
                SetStrategyKind::Empty
            );
            assert!((*(dst as *const W_SetObject)).sstorage.is_null());
            assert_eq!(w_set_len(dst), 0);

            assert!(w_set_difference_update_from_set(other, empty2).is_ok());
            assert_eq!(w_set_len(other), 1);
            assert!(w_set_difference_update_from_set(empty2, other).is_ok());
            assert_eq!(
                (*(empty2 as *const W_SetObject)).sstrategy.kind,
                SetStrategyKind::Empty
            );
            assert!((*(empty2 as *const W_SetObject)).sstorage.is_null());

            // `AbstractUnwrappedSetStrategy.popitem` keeps the strategy when
            // the dict underneath becomes empty.
            let popped = w_set_popitem(other);
            assert!(popped.is_some());
            assert_eq!(w_set_len(other), 0);
            assert_eq!(
                (*(other as *const W_SetObject)).sstrategy.kind,
                SetStrategyKind::Int
            );
            assert!(!(*(other as *const W_SetObject)).sstorage.is_null());
            // Same storage (`s -= s`) goes through `w_set_clear`.
            assert!(w_set_difference_update_from_set(other, other).is_ok());
            assert_eq!(
                (*(other as *const W_SetObject)).sstrategy.kind,
                SetStrategyKind::Empty
            );
            assert!((*(other as *const W_SetObject)).sstorage.is_null());
        }
    }

    #[test]
    fn integer_strategy_switches_to_object_without_renumbering_slots() {
        install_test_hash_hook();
        unsafe {
            let s = w_set_new();
            w_set_add(s, w_int_new(1));
            assert_eq!(
                (*(s as *const W_SetObject)).sstrategy.kind,
                SetStrategyKind::Int
            );
            w_set_add(s, w_int_new(2));
            assert_eq!(
                (*(s as *const W_SetObject)).sstrategy.kind,
                SetStrategyKind::Int
            );
            assert!(w_set_contains(s, w_int_new(1)));
            assert!(!w_set_contains(s, w_int_new(99)));
            w_set_add(s, w_int_new(3));
            assert!(w_set_discard(s, w_int_new(2)));
            let slot_a = w_set_next_slot(s, 0).unwrap();
            let key_a = w_set_key_at(s, slot_a).unwrap();
            assert_eq!(crate::w_int_get_value(key_a.obj), 1);
            assert_eq!(key_a.hash, 1);
            let slot_b = w_set_next_slot(s, slot_a + 1).unwrap();
            let key_b = w_set_key_at(s, slot_b).unwrap();
            assert_eq!(crate::w_int_get_value(key_b.obj), 3);
            let hole = slot_a + 1;
            if hole != slot_b {
                assert!(w_set_key_at(s, hole).is_none());
            }
            w_set_add(s, crate::w_str_new("x"));
            assert_eq!(
                (*(s as *const W_SetObject)).sstrategy.kind,
                SetStrategyKind::Object
            );
            assert_eq!(w_set_len(s), 3);
            assert_eq!(
                crate::w_int_get_value(w_set_key_at(s, slot_a).unwrap().obj),
                1
            );
            assert_eq!(
                crate::w_int_get_value(w_set_key_at(s, slot_b).unwrap().obj),
                3
            );
            if hole != slot_b {
                assert!(w_set_key_at(s, hole).is_none());
            }
            assert!(w_set_contains(s, w_int_new(1)));
            assert!(w_set_contains(s, w_int_new(3)));
            assert!(w_set_contains(s, crate::w_str_new("x")));
            assert!(!w_set_contains(s, w_int_new(2)));

            let t = w_set_new();
            w_set_add(t, w_int_new(5));
            assert!(w_set_discard(t, w_int_new(5)));
            assert_eq!(
                (*(t as *const W_SetObject)).sstrategy.kind,
                SetStrategyKind::Empty
            );
            assert!((*(t as *const W_SetObject)).sstorage.is_null());

            let p = w_set_new();
            w_set_add(p, w_int_new(9));
            let popped = w_set_popitem(p).unwrap();
            assert!(crate::is_int(popped));
            assert_eq!(crate::w_int_get_value(popped), 9);

            // `is_plain_int1` rejects bool, so `{True}` stays an object set.
            let b = w_set_new();
            w_set_add(b, crate::w_bool_from(true));
            assert_eq!(
                (*(b as *const W_SetObject)).sstrategy.kind,
                SetStrategyKind::Object
            );
        }
    }

    #[test]
    fn w_set_gc_type_id_matches_descr() {
        assert_eq!(W_SET_GC_TYPE_ID, 30);
        assert_eq!(
            <W_SetObject as crate::lltype::GcType>::type_id(),
            W_SET_GC_TYPE_ID
        );
        assert_eq!(
            <W_SetObject as crate::lltype::GcType>::SIZE,
            W_SET_OBJECT_SIZE
        );
    }
}
