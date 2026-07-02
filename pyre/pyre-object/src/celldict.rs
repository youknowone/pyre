//! `pypy/objspace/std/celldict.py` port — module / globals dict
//! strategy backed by a version-tagged map of `str -> value_or_cell`.
//!
//! Header docstring at upstream `celldict.py:1-4`:
//!
//! ```text
//! A very simple cell dict implementation using a version tag.
//! The dictionary maps keys to objects. If a specific key is
//! changed a lot, a level of indirection is introduced to make
//! the version tag change less often.
//! ```
//!
//! The `ModuleDictStrategy` struct itself
//! plus the supporting `VersionTag` / `ModuleDictStorage` types.
//! Bodies are stubbed against an in-memory `Vec<(String,
//! PyObjectRef)>` because the cell-indirection layer
//! (`pypy/objspace/std/typeobject.py:22-71 MutableCell / write_cell
//! / unwrap_cell`) is a separate port.  Until that lands the
//! strategy stores raw values directly, which is observationally
//! correct (cells are a JIT speed optimisation, not a semantic
//! requirement).
//!
//! `W_ModuleDictObject` (`dictmultiobject.rs`) carries this strategy
//! as its `mstrategy` slot per `dictmultiobject.py:328-341`.  The
//! trait `impl crate::dictmultiobject::DictStrategy for
//! ModuleDictStrategy` lives at the bottom of this file and routes
//! every method to the existing `w_module_dict_*` / `w_dict_*`
//! free functions so callers can dispatch polymorphically via
//! `w_dict_get_strategy(obj)`.

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)]

use crate::pyobject::*;
use crate::w_str_new;

// ── MutableCell family ──────────────────────────────────────────────
//
// `pypy/objspace/std/typeobject.py:22-71` defines the cell layer
// referenced by `celldict.py:80-90 _setitem_str_cell_known` and
// `:143-145 getitem_str`.  PyPy keeps a level of indirection so that
// frequently-rewritten module / type attributes mutate the cell's
// payload without bumping `mstrategy.version`, which keeps JIT inline
// caches keyed on that version valid across the rewrite.
//
//     class MutableCell(W_Root):
//         def unwrap_cell(self, space): raise NotImplementedError
//
//     class ObjectMutableCell(MutableCell):
//         def __init__(self, w_value=None):
//             self.w_value = w_value
//         def unwrap_cell(self, space):
//             return self.w_value
//
//     class IntMutableCell(MutableCell):
//         def __init__(self, intvalue):
//             self.intvalue = intvalue
//         def unwrap_cell(self, space):
//             return space.newint(self.intvalue)
//
//     def unwrap_cell(space, w_value):
//         if isinstance(w_value, MutableCell):
//             return w_value.unwrap_cell(space)
//         return w_value
//
//     def write_cell(space, w_cell, w_value):
//         if w_cell is None:
//             return w_value
//         if isinstance(w_cell, ObjectMutableCell):
//             w_cell.w_value = w_value
//             return None
//         elif isinstance(w_cell, IntMutableCell) and is_plain_int1(w_value):
//             w_cell.intvalue = plain_int_w(space, w_value)
//             return None
//         elif space.is_w(w_cell, w_value):
//             return None
//         if is_plain_int1(w_value):
//             return IntMutableCell(plain_int_w(space, w_value))
//         else:
//             return ObjectMutableCell(w_value)

/// Internal type tag for `ObjectMutableCell`.  Never user-visible —
/// cells live inside the module dict's storage and are unwrapped
/// before any read crosses out to user code.  The PyType is allocated
/// so that `py_type_check` can disambiguate cells from real values
/// without a separate type-id field.
pub static OBJECT_MUTABLE_CELL_TYPE: PyType = new_pytype("__ObjectMutableCell");

/// Internal type tag for `IntMutableCell`.
pub static INT_MUTABLE_CELL_TYPE: PyType = new_pytype("__IntMutableCell");

/// GC type id assigned to `ObjectMutableCell` — slot 49, immediately
/// after `W_MODULE_DICT_GC_TYPE_ID=48`.
pub const W_OBJECT_MUTABLE_CELL_GC_TYPE_ID: u32 = 49;

/// GC type id assigned to `IntMutableCell`.
pub const W_INT_MUTABLE_CELL_GC_TYPE_ID: u32 = 50;

/// `typeobject.py:26-34 ObjectMutableCell`.
#[repr(C)]
pub struct ObjectMutableCell {
    pub ob_header: PyObject,
    pub w_value: PyObjectRef,
}

/// `typeobject.py:37-45 IntMutableCell`.
#[repr(C)]
pub struct IntMutableCell {
    pub ob_header: PyObject,
    pub intvalue: i64,
}

pub const W_OBJECT_MUTABLE_CELL_OBJECT_SIZE: usize = std::mem::size_of::<ObjectMutableCell>();
pub const W_INT_MUTABLE_CELL_OBJECT_SIZE: usize = std::mem::size_of::<IntMutableCell>();

/// Byte offset of the inline `PyObjectRef` field the GC must trace
/// during minor collection.  Mirrors `W_CELL_GC_PTR_OFFSETS` on the
/// closure-cell layer (`nestedscope.rs:42`).
pub const W_OBJECT_MUTABLE_CELL_GC_PTR_OFFSETS: [usize; 1] =
    [std::mem::offset_of!(ObjectMutableCell, w_value)];

impl crate::lltype::GcType for ObjectMutableCell {
    fn type_id() -> u32 {
        W_OBJECT_MUTABLE_CELL_GC_TYPE_ID
    }
    const SIZE: usize = W_OBJECT_MUTABLE_CELL_OBJECT_SIZE;
}

impl crate::lltype::GcType for IntMutableCell {
    fn type_id() -> u32 {
        W_INT_MUTABLE_CELL_GC_TYPE_ID
    }
    const SIZE: usize = W_INT_MUTABLE_CELL_OBJECT_SIZE;
}

/// `typeobject.py:27-28 ObjectMutableCell.__init__`.
pub fn w_object_mutable_cell_new(w_value: PyObjectRef) -> PyObjectRef {
    crate::lltype::malloc_typed(ObjectMutableCell {
        ob_header: PyObject {
            ob_type: &OBJECT_MUTABLE_CELL_TYPE as *const PyType,
            w_class: get_instantiate(&OBJECT_MUTABLE_CELL_TYPE),
        },
        w_value,
    }) as PyObjectRef
}

/// `typeobject.py:38-39 IntMutableCell.__init__`.
pub fn w_int_mutable_cell_new(intvalue: i64) -> PyObjectRef {
    crate::lltype::malloc_typed(IntMutableCell {
        ob_header: PyObject {
            ob_type: &INT_MUTABLE_CELL_TYPE as *const PyType,
            w_class: get_instantiate(&INT_MUTABLE_CELL_TYPE),
        },
        intvalue,
    }) as PyObjectRef
}

/// `isinstance(w, ObjectMutableCell)` predicate.
///
/// # Safety
/// `obj` must be a valid non-null PyObjectRef.
#[inline]
pub unsafe fn is_object_mutable_cell(obj: PyObjectRef) -> bool {
    !obj.is_null() && py_type_check(obj, &OBJECT_MUTABLE_CELL_TYPE)
}

/// `isinstance(w, IntMutableCell)` predicate.
///
/// # Safety
/// `obj` must be a valid non-null PyObjectRef.
#[inline]
pub unsafe fn is_int_mutable_cell(obj: PyObjectRef) -> bool {
    !obj.is_null() && py_type_check(obj, &INT_MUTABLE_CELL_TYPE)
}

/// `isinstance(w, MutableCell)`.
#[inline]
pub unsafe fn is_mutable_cell(obj: PyObjectRef) -> bool {
    is_object_mutable_cell(obj) || is_int_mutable_cell(obj)
}

/// `typeobject.py:48-51 unwrap_cell`:
///
/// ```python
/// def unwrap_cell(space, w_value):
///     if isinstance(w_value, MutableCell):
///         return w_value.unwrap_cell(space)
///     return w_value
/// ```
///
/// Hot path: read `ob_type` once, two pointer-equality compares
/// against the two static cell type tags.  The common case is a
/// non-cell value (raw function / int / etc.), so both compares
/// fall through to the identity return without further memory traffic.
///
/// # Safety
/// `w_value` must be a valid PyObjectRef (null tolerated).
#[inline]
pub unsafe fn unwrap_cell(w_value: PyObjectRef) -> PyObjectRef {
    if w_value.is_null() {
        return w_value;
    }
    let tp = (*w_value).ob_type;
    if std::ptr::eq(tp, &OBJECT_MUTABLE_CELL_TYPE as *const PyType) {
        return (*(w_value as *const ObjectMutableCell)).w_value;
    }
    if std::ptr::eq(tp, &INT_MUTABLE_CELL_TYPE as *const PyType) {
        return crate::w_int_new((*(w_value as *const IntMutableCell)).intvalue);
    }
    w_value
}

/// Forward the single movable `PyObjectRef` reachable through a module
/// dict value slot during a GC root walk.
///
/// A module dict entry (and a `GlobalCache.cell`) is either a raw
/// `w_value` or a `MutableCell` wrapping it (`typeobject.py:48-51
/// unwrap_cell`).  `MutableCell`s are `malloc_typed`
/// (`w_object_mutable_cell_new` / `w_int_mutable_cell_new`), so the
/// collector never relocates the cell itself and never recurses into
/// it; for an `ObjectMutableCell` the inner `w_value` is the movable
/// reference that must be forwarded in place, while an `IntMutableCell`
/// holds an unboxed `i64` with no reference to forward.  A non-cell slot
/// holds the movable value directly and is forwarded as-is.
///
/// # Safety
/// `slot` must point to a valid `PyObjectRef` (null tolerated).
pub unsafe fn walk_module_value_slot(
    slot: &mut PyObjectRef,
    visitor: &mut dyn FnMut(&mut PyObjectRef),
) {
    let w_value = *slot;
    if w_value.is_null() {
        return;
    }
    let tp = (*w_value).ob_type;
    if std::ptr::eq(tp, &OBJECT_MUTABLE_CELL_TYPE as *const PyType) {
        let cell = &mut *(w_value as *mut ObjectMutableCell);
        visitor(&mut cell.w_value);
    } else if std::ptr::eq(tp, &INT_MUTABLE_CELL_TYPE as *const PyType) {
        // IntMutableCell carries an unboxed i64; no GC reference.
    } else {
        visitor(slot);
    }
}

/// `typeobject.py:53-71 write_cell`:
///
/// ```python
/// def write_cell(space, w_cell, w_value):
///     if w_cell is None:
///         return w_value
///     if isinstance(w_cell, ObjectMutableCell):
///         w_cell.w_value = w_value
///         return None
///     elif isinstance(w_cell, IntMutableCell) and is_plain_int1(w_value):
///         w_cell.intvalue = plain_int_w(space, w_value)
///         return None
///     elif space.is_w(w_cell, w_value):
///         return None
///     if is_plain_int1(w_value):
///         return IntMutableCell(plain_int_w(space, w_value))
///     else:
///         return ObjectMutableCell(w_value)
/// ```
///
/// `Option<PyObjectRef>` return: `None` => the cell mutation was
/// in-place; the storage's existing entry stays.  `Some(w)` => the
/// caller must write `w` into the storage slot (either a brand-new
/// cell or the raw value for the no-cell-yet case).
///
/// # Safety
/// `w_cell` must be either `None` or a valid PyObjectRef.  `w_value`
/// must be a valid non-null PyObjectRef.
pub unsafe fn write_cell(w_cell: Option<PyObjectRef>, w_value: PyObjectRef) -> Option<PyObjectRef> {
    // The cell payload lives in a Box-immortal structure reached only by the
    // prebuilt-family root walk; record the store so the next minor
    // collection rescans it (gc_roots.rs prebuilt-root write tracking).
    crate::gc_roots::mark_prebuilt_roots_dirty();
    let Some(w_cell) = w_cell else {
        // attribute does not exist at all, write it without a cell first
        return Some(w_value);
    };
    if is_object_mutable_cell(w_cell) {
        (*(w_cell as *mut ObjectMutableCell)).w_value = w_value;
        return None;
    }
    if is_int_mutable_cell(w_cell) && crate::listobject::is_plain_int1(w_value) {
        (*(w_cell as *mut IntMutableCell)).intvalue = crate::listobject::plain_int_w(w_value);
        return None;
    }
    // If the new value and the current value are the same, don't
    // create a level of indirection, or mutate the version.
    if std::ptr::eq(w_cell, w_value) {
        return None;
    }
    if crate::listobject::is_plain_int1(w_value) {
        return Some(w_int_mutable_cell_new(crate::listobject::plain_int_w(
            w_value,
        )));
    }
    Some(w_object_mutable_cell_new(w_value))
}

/// `pypy/objspace/std/celldict.py:20-21 VersionTag`:
///
/// ```python
/// class VersionTag(object):
///     pass
/// ```
///
/// An opaque identity tag invalidated on every mutation that affects
/// the JIT's view of the dict.  Pyre's stand-in is a monotonically
/// increasing counter — pointer-identity matches PyPy's `is` test
/// because each `Box<VersionTag>` allocates a fresh address but a
/// counter is JIT-friendlier and trivially `Copy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VersionTag(pub u64);

impl VersionTag {
    /// Allocate a fresh, never-before-seen version tag.
    pub fn fresh() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(1);
        VersionTag(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

/// `pypy/objspace/std/celldict.py:24-25 _wrapkey`:
///
/// ```python
/// def _wrapkey(space, key):
///     return space.newtext(key)
/// ```
///
/// Wraps a Rust `&str` key as a Python `str` PyObjectRef.
#[inline]
pub fn _wrapkey(key: &str) -> PyObjectRef {
    w_str_new(key)
}

/// Strategy-owned storage for `ModuleDictStrategy`.
///
/// `celldict.py:30-31,41-42`:
///
/// ```python
/// erase, unerase = rerased.new_erasing_pair("modulecell")
/// ...
/// def get_empty_storage(self):
///     return self.erase({})
/// ```
///
/// PyPy erases a real Python `{}` dict — insertion-ordered (since
/// Python 3.7) and O(1) hashed.  Pyre's port wraps
/// `indexmap::IndexMap<String, PyObjectRef>`, which provides the same
/// insertion-ordered + hashed semantics directly so the strategy
/// contract on `:188-198 getiter{keys,values,items,reversed}`
/// continues to honour insertion order while `get` / `set` / `remove`
/// stay O(1) amortised.
#[derive(Default)]
pub struct ModuleDictStorage {
    pub entries: indexmap::IndexMap<String, PyObjectRef>,
}

impl ModuleDictStorage {
    pub fn new() -> Self {
        Self {
            entries: indexmap::IndexMap::new(),
        }
    }

    /// `dict.__len__`.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `dict.__getitem__(key)` returning the raw stored value (or
    /// cell — eventually).  None when absent.
    pub fn get(&self, key: &str) -> Option<PyObjectRef> {
        self.entries.get(key).copied()
    }

    /// `dict[key] = w_value` — insertion-ordered.  Returns the
    /// previous value (or None if this is a fresh slot).
    pub fn set(&mut self, key: &str, w_value: PyObjectRef) -> Option<PyObjectRef> {
        // Prebuilt-family store (see `write_cell`): the module-dict storage
        // is Box-immortal, so the write-tracking bit is its write barrier.
        crate::gc_roots::mark_prebuilt_roots_dirty();
        // `IndexMap::insert` preserves the existing slot's position
        // on overwrite, matching Python `{}`'s assignment semantics
        // (rewriting an existing key does not move it to the end).
        self.entries.insert(key.to_string(), w_value)
    }

    /// `del dict[key]` — returns the removed value or None.
    ///
    /// Uses `shift_remove` (not `swap_remove`) so the remaining keys
    /// keep their relative insertion order, matching Python `dict`'s
    /// `__delitem__` semantics that `celldict.py:159-160 items` /
    /// `:166-171 popitem` (LIFO) depend on.
    pub fn remove(&mut self, key: &str) -> Option<PyObjectRef> {
        self.entries.shift_remove(key)
    }

    /// `dict.clear()`.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// `&mut PyObjectRef` iterator over every stored value, for the
    /// GC custom-trace hook on W_ModuleDictObject.
    pub fn iter_values_mut(&mut self) -> impl Iterator<Item = &mut PyObjectRef> + '_ {
        self.entries.values_mut()
    }
}

/// `pypy/objspace/std/celldict.py:261-277 GlobalCache`:
///
/// ```python
/// class GlobalCache(object):
///     def __init__(self, cell):
///         # works like this: self.cell is always the result of
///         # getdictvalue_no_unwrapping on the equivalent key.
///         # this means it is None if the key doesn't exist, a w_value if there is
///         # no cell, or a Cell
///         #
///         # if the module dict actually switches to a different strategy, then
///         # cell is set to None, and valid to False
///         self.cell = cell
///         self.valid = True
///         self.ref = weakref.ref(self)
///         self.builtincache = None
///
///     @objectmodel.always_inline
///     def getvalue(self, space):
///         return unwrap_cell(space, self.cell)
/// ```
///
/// Per-name fast-path cache for `LOAD_GLOBAL`.  `cell` is the result
/// of `getdictvalue_no_unwrapping` at cache install time; `valid`
/// flips to `false` when the strategy invalidates (mutated() or
/// switch_to_object_strategy).  `builtincache` chains a nested cache
/// for the builtins fallback so that `LOAD_GLOBAL` of a name that
/// lives in builtins still hits one indirection rather than
/// re-traversing through `__builtins__.w_dict` every call.
pub struct GlobalCache {
    pub cell: Option<PyObjectRef>,
    pub valid: bool,
    /// `celldict.py:235 cache.builtincache = builtincache`: stores the
    /// _same_ `GlobalCache` object that lives inside the builtin
    /// strategy's `caches` map, so a write through the builtin's
    /// strategy that updates `cache.cell` is immediately visible
    /// here.  PyPy stores this as a direct Python attribute
    /// (ref-counted strong ref); pyre uses `Rc<RefCell<...>>` so the
    /// cache stays alive as long as either the owning strategy's
    /// `caches` map OR a chained `builtincache` holds it — matching
    /// PyPy's ref-counted lifetime against
    /// `ModuleDictStrategy.invalidate_caches` dropping the registry.
    pub builtincache: Option<std::rc::Rc<std::cell::RefCell<GlobalCache>>>,
}

impl GlobalCache {
    pub fn new(cell: Option<PyObjectRef>) -> Self {
        Self {
            cell,
            valid: true,
            builtincache: None,
        }
    }

    /// `celldict.py:275-277 getvalue`: return the cached cell's
    /// unwrapped value, or `None` if the cache holds `None` (key
    /// absent at install time).
    #[inline]
    pub unsafe fn getvalue(&self) -> Option<PyObjectRef> {
        self.cell.map(|c| unwrap_cell(c))
    }
}

/// GC root walk over a single `GlobalCache` and its chained
/// `builtincache`, forwarding the movable value each `cell` holds.
///
/// Mirrors the `GlobalCache` object graph shape (`celldict.py:261-277`):
/// `cell` plus the nested `builtincache`, which always points at a
/// DIFFERENT `GlobalCache` instance (`celldict.py:236
/// builtin_strategy.get_global_cache(...)`), so the outer borrow is
/// dropped before recursing.  The cached `cell` duplicates the storage
/// entry it was installed from, so a collection that relocates that
/// value must rewrite the cache copy too or the next cached
/// `LOAD_GLOBAL` reads a stale pointer.
///
/// # Safety
/// `cache.cell`, when present, must be a valid `PyObjectRef`.
unsafe fn walk_one_global_cache(
    cache: &std::rc::Rc<std::cell::RefCell<GlobalCache>>,
    visitor: &mut dyn FnMut(&mut PyObjectRef),
) {
    let mut c = cache.borrow_mut();
    if let Some(slot) = c.cell.as_mut() {
        walk_module_value_slot(slot, visitor);
    }
    if let Some(builtincache) = c.builtincache.clone() {
        drop(c);
        walk_one_global_cache(&builtincache, visitor);
    }
}

/// `pypy/objspace/std/celldict.py:28-240 ModuleDictStrategy`.
///
/// ```python
/// class ModuleDictStrategy(DictStrategy):
///     erase, unerase = rerased.new_erasing_pair("modulecell")
///     erase = staticmethod(erase)
///     unerase = staticmethod(unerase)
///
///     _immutable_fields_ = ["version?"]
///
///     def __init__(self, space):
///         self.space = space
///         self.version = VersionTag()
///         self.caches = None
/// ```
///
/// Pyre's port omits the `erase / unerase` static methods because
/// Rust's strong typing makes the rerasure unnecessary —
/// `ModuleDictStorage` is the concrete storage type directly.
///
/// `caches` is the per-name `GlobalCache` registry consulted by the
/// `LOAD_GLOBAL` fast path (`celldict.py:214 get_global_cache`).
/// Allocated lazily on first cache install.
pub struct ModuleDictStrategy {
    pub version: VersionTag,
    pub caches:
        Option<std::collections::HashMap<String, std::rc::Rc<std::cell::RefCell<GlobalCache>>>>,
    /// JIT loop-invalidation flags watching the `version?` quasi-immutable
    /// field.  Each compiled loop whose trace promoted `self.version`
    /// (and folded a module-global lookup keyed on it) registers its
    /// `JitCellToken` invalidation flag here.  `mutated()` reassigns
    /// `version`, which under `_immutable_fields_ = ["version?"]` must
    /// invalidate every such loop, so it flips all live flags.  Weak refs
    /// so a dead loop token drops out without keeping the flag alive.
    version_watchers: Vec<std::sync::Weak<std::sync::atomic::AtomicBool>>,
}

impl Default for ModuleDictStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleDictStrategy {
    /// `celldict.py:36-39 __init__`.
    pub fn new() -> Self {
        Self {
            version: VersionTag::fresh(),
            caches: None,
            version_watchers: Vec::new(),
        }
    }

    /// Register a JIT loop's invalidation flag against the `version?`
    /// quasi-immutable field.  The compile-time glue
    /// (`register_quasi_immutable_deps` analogue) calls this once per
    /// version-keyed module-global dependency, passing the
    /// `JitCellToken.invalidation_flag()`.  Mirrors
    /// `DictStorage::register_slot_watcher` but keyed on the strategy
    /// version rather than a per-slot index.
    pub fn register_version_watcher(
        &mut self,
        flag: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) {
        self.version_watchers.push(std::sync::Arc::downgrade(flag));
    }

    /// Invalidate every loop watching `version`.  Sets each live flag to
    /// `true` (the polarity `GuardNotInvalidated` tests) and drops dead
    /// weak refs.  Mirrors `DictStorage::notify_slot_watchers`.
    fn notify_version_watchers(&mut self) {
        if self.version_watchers.is_empty() {
            return;
        }
        self.version_watchers.retain(|w| {
            if let Some(flag) = w.upgrade() {
                flag.store(true, std::sync::atomic::Ordering::Release);
                true
            } else {
                false
            }
        });
    }

    /// `celldict.py:214-240 get_global_cache`:
    ///
    /// ```python
    /// def get_global_cache(self, w_dict, key):
    ///     space = w_dict.space
    ///     if self.caches is None:
    ///         cache = None
    ///         self.caches = {}
    ///     else:
    ///         cache = self.caches.get(key, None)
    ///     if cache is None:
    ///         cell = self.getdictvalue_no_unwrapping(w_dict, key)
    ///         cache = GlobalCache(cell)
    ///         if (not space.config.objspace.honor__builtins__ and
    ///                 cell is None and
    ///                 w_dict is not space.builtin.w_dict):
    ///             # …attach `cache.builtincache` …
    ///         self.caches[key] = cache
    ///     return cache
    /// ```
    ///
    /// Pyre's `space` analogue always picks the builtin per frame
    /// (`PyFrame.w_builtin` assigned at construction, mirroring
    /// `pyframe.py:115 self.builtin = space.builtin.pick_builtin
    /// (w_globals)` under `honor__builtins__=True`).  Per
    /// `celldict.py:224 not space.config.objspace.honor__builtins__`
    /// the builtincache install is therefore a no-op — attaching a
    /// cache keyed to `space.builtin.w_dict` would mis-fire whenever
    /// a frame's picked builtin differs from the singleton.  Only the
    /// per-dict cell cache is installed here; the builtin lookup
    /// stays a live `space.finditem_str(frame.w_builtin.w_dict, name)`
    /// at every call (see `load_global_via_cache` final fallback in
    /// `pyre-interpreter/src/eval.rs`).
    pub fn get_global_cache(
        &mut self,
        storage: &ModuleDictStorage,
        key: &str,
    ) -> std::rc::Rc<std::cell::RefCell<GlobalCache>> {
        if self.caches.is_none() {
            self.caches = Some(std::collections::HashMap::new());
        }
        let already_present = match self.caches.as_ref() {
            Some(c) => c.contains_key(key),
            None => false,
        };
        if already_present {
            return self.caches.as_ref().unwrap().get(key).unwrap().clone();
        }
        let cell = self.getdictvalue_no_unwrapping(storage, key);
        let caches = self.caches.as_mut().unwrap();
        // `celldict.py:223 cache = GlobalCache(cell)`.  Lines 224-238
        // (`if not honor__builtins__ and cell is None and w_dict is
        // not space.builtin.w_dict:` …) are skipped because pyre is
        // permanently in `honor__builtins__=True` mode (see method
        // docstring above); builtincache attachment is unreachable.
        // The fresh cache duplicates the (possibly nursery-young) cell /
        // value pointer into walker-only storage (`walk_cache_cells`);
        // record the store like any other prebuilt-family write.
        crate::gc_roots::mark_prebuilt_roots_dirty();
        let rc = std::rc::Rc::new(std::cell::RefCell::new(GlobalCache::new(cell)));
        caches.insert(key.to_string(), rc.clone());
        rc
    }

    /// `celldict.py:180-184 switch_to_object_strategy` cache flush:
    ///
    /// ```python
    /// if self.caches is not None:
    ///     for cache in self.caches.itervalues():
    ///         cache.cell = None
    ///         cache.valid = False
    ///     self.caches = None
    /// ```
    ///
    /// Invalidate every live cache and drop the registry.  Called from
    /// the host `switch_to_object_strategy` helper on
    /// W_ModuleDictObject.
    pub fn invalidate_caches(&mut self) {
        if let Some(caches) = self.caches.as_mut() {
            for cache in caches.values() {
                let mut c = cache.borrow_mut();
                c.cell = None;
                c.valid = false;
                c.builtincache = None;
            }
        }
        self.caches = None;
    }

    /// `celldict.py:41-42 get_empty_storage`:
    ///
    /// ```python
    /// def get_empty_storage(self):
    ///     return self.erase({})
    /// ```
    pub fn get_empty_storage(&self) -> ModuleDictStorage {
        ModuleDictStorage::new()
    }

    /// `celldict.py:44-45 mutated`:
    ///
    /// ```python
    /// def mutated(self):
    ///     self.version = VersionTag()
    /// ```
    ///
    /// Reassigning the `version?` quasi-immutable field is what
    /// invalidates the JIT.  In RPython the `?` machinery does this
    /// automatically on the field write; pyre flips the registered loop
    /// flags explicitly here (the single `version` write site).
    #[inline]
    pub fn mutated(&mut self) {
        self.version = VersionTag::fresh();
        self.notify_version_watchers();
    }

    /// `celldict.py:47-55 getdictvalue_no_unwrapping`:
    ///
    /// ```python
    /// def getdictvalue_no_unwrapping(self, w_dict, key):
    ///     self = jit.promote(self)
    ///     return self._getdictvalue_no_unwrapping_pure(
    ///         self.version, w_dict, key)
    ///
    /// @jit.elidable_promote('0,1,2')
    /// def _getdictvalue_no_unwrapping_pure(self, version, w_dict, key):
    ///     return self.unerase(w_dict.dstorage).get(key, None)
    /// ```
    ///
    /// Returns the raw stored value (in PyPy this would be a
    /// `MutableCell` or a plain `PyObjectRef`; pyre stores plain
    /// values until the cell-indirection slice lands).
    pub fn getdictvalue_no_unwrapping(
        &self,
        storage: &ModuleDictStorage,
        key: &str,
    ) -> Option<PyObjectRef> {
        storage.get(key)
    }

    /// `celldict.py:76-78 setitem_str`:
    ///
    /// ```python
    /// def setitem_str(self, w_dict, key, w_value):
    ///     cell = self.getdictvalue_no_unwrapping(w_dict, key)
    ///     return self._setitem_str_cell_known(cell, w_dict, key, w_value)
    /// ```
    pub fn setitem_str(
        &mut self,
        storage: &mut ModuleDictStorage,
        key: &str,
        w_value: PyObjectRef,
    ) {
        let cell = self.getdictvalue_no_unwrapping(storage, key);
        self._setitem_str_cell_known(cell, storage, key, w_value);
    }

    /// `celldict.py:80-90 _setitem_str_cell_known`:
    ///
    /// ```python
    /// def _setitem_str_cell_known(self, cell, w_dict, key, w_value):
    ///     w_value = write_cell(self.space, cell, w_value)
    ///     if w_value is None:
    ///         return
    ///     self.mutated()
    ///     self.unerase(w_dict.dstorage)[key] = w_value
    ///     if self.caches is None:
    ///         return
    ///     cache = self.caches.get(key, None)
    ///     if cache:
    ///         cache.cell = w_value
    /// ```
    ///
    /// Cell-indirection (`write_cell`) is stubbed to identity until
    /// the `MutableCell` family ports.  Without cells, every write
    /// reaches `storage[key] = w_value` and triggers `mutated()`.
    pub fn _setitem_str_cell_known(
        &mut self,
        cell: Option<PyObjectRef>,
        storage: &mut ModuleDictStorage,
        key: &str,
        w_value: PyObjectRef,
    ) {
        let Some(w_to_store) = (unsafe { write_cell(cell, w_value) }) else {
            // In-place cell mutation: storage slot unchanged, version
            // stays valid (matches the JIT-cache-stable fast path).
            return;
        };
        self.mutated();
        storage.set(key, w_to_store);
        // `celldict.py:88-90`: keep any live cache for `key` in step
        // with the new stored value so subsequent LOAD_GLOBAL through
        // the cache reads the fresh entry without an invalidation
        // round-trip.
        if let Some(caches) = self.caches.as_mut() {
            if let Some(cache) = caches.get(key) {
                cache.borrow_mut().cell = Some(w_to_store);
            }
        }
    }

    /// `celldict.py:128-129 length`:
    ///
    /// ```python
    /// def length(self, w_dict):
    ///     return len(self.unerase(w_dict.dstorage))
    /// ```
    pub fn length(&self, storage: &ModuleDictStorage) -> usize {
        storage.len()
    }

    /// `celldict.py:143-145 getitem_str`:
    ///
    /// ```python
    /// def getitem_str(self, w_dict, key):
    ///     cell = self.getdictvalue_no_unwrapping(w_dict, key)
    ///     return unwrap_cell(self.space, cell)
    /// ```
    pub fn getitem_str(&self, storage: &ModuleDictStorage, key: &str) -> Option<PyObjectRef> {
        let raw = self.getdictvalue_no_unwrapping(storage, key)?;
        Some(unsafe { unwrap_cell(raw) })
    }

    /// `celldict.py:106-126 delitem` — minimal str-key path
    /// (`celldict.py:110-121`); the object-fallback /
    /// `_never_equal_to_string` branches belong to the full strategy
    /// dispatch once `ObjectDictStrategy` is wired.
    pub fn delitem_str(
        &mut self,
        storage: &mut ModuleDictStorage,
        key: &str,
    ) -> Option<PyObjectRef> {
        let removed = storage.remove(key)?;
        if let Some(caches) = self.caches.as_mut() {
            if let Some(cache) = caches.get(key) {
                // `celldict.py:117-121`: zero out the per-key cache
                // so LOAD_GLOBAL falls through to the builtins
                // fallback (or NameError) on the next read.
                cache.borrow_mut().cell = None;
            }
        }
        self.mutated();
        Some(removed)
    }

    /// `celldict.py:162-164 clear`:
    ///
    /// ```python
    /// def clear(self, w_dict):
    ///     self.unerase(w_dict.dstorage).clear()
    ///     self.mutated()
    /// ```
    pub fn clear(&mut self, storage: &mut ModuleDictStorage) {
        storage.clear();
        self.mutated();
    }

    /// `celldict.py:188-189 getiterkeys`:
    ///
    /// ```python
    /// def getiterkeys(self, w_dict):
    ///     return self.unerase(w_dict.dstorage).iterkeys()
    /// ```
    pub fn getiterkeys<'a>(
        &self,
        storage: &'a ModuleDictStorage,
    ) -> impl Iterator<Item = &'a str> + 'a {
        storage.entries.keys().map(|k| k.as_str())
    }

    /// `celldict.py:191-192 getitervalues`:
    ///
    /// ```python
    /// def getitervalues(self, w_dict):
    ///     return self.unerase(w_dict.dstorage).itervalues()
    /// ```
    ///
    /// The skeleton omits the per-element `unwrap_cell` because no
    /// cells are stored yet (see `_setitem_str_cell_known`).
    pub fn getitervalues<'a>(
        &self,
        storage: &'a ModuleDictStorage,
    ) -> impl Iterator<Item = PyObjectRef> + 'a {
        // `celldict.py:152-154 values`: each cell is unwrapped before
        // it crosses out of the strategy.  Without unwrapping, JIT-
        // promoted cell objects would leak into user space and break
        // identity comparisons against the previously-stored value.
        storage.entries.values().map(|v| unsafe { unwrap_cell(*v) })
    }

    /// GC root walk over every live `GlobalCache.cell` reachable through
    /// this strategy's `caches` registry (`celldict.py:214
    /// get_global_cache`), forwarding the movable value each cell holds.
    ///
    /// # Safety
    /// Each cached `GlobalCache.cell`, when present, must be a valid
    /// `PyObjectRef`.
    pub unsafe fn walk_cache_cells(&self, visitor: &mut dyn FnMut(&mut PyObjectRef)) {
        if let Some(caches) = self.caches.as_ref() {
            for cache in caches.values() {
                walk_one_global_cache(cache, visitor);
            }
        }
    }
}

/// `pypy/objspace/std/celldict.py:28 ModuleDictStrategy(DictStrategy)`
/// — abstract base inheritance.  Every method takes the `W_ModuleDict
/// Object` (`w_dict: PyObjectRef`) and resolves its `dstorage`
/// internally, matching PyPy's strategy contract per
/// `dictmultiobject.py:462 DictStrategy`.
///
/// The pyre inherent methods on `ModuleDictStrategy`
/// (`setitem_str(&self, &mut ModuleDictStorage, …)` etc.) are a
/// pre-existing pyre adaptation that pairs strategy with storage
/// directly — kept for `celldict::tests` plus a handful of legacy
/// callers, but the canonical surface going forward is the
/// trait dispatch below.
impl crate::dictmultiobject::DictStrategy for ModuleDictStrategy {
    fn strategy_kind(&self) -> crate::dictmultiobject::StrategyKind {
        crate::dictmultiobject::StrategyKind::Module
    }

    /// `celldict.py:46-49 get_empty_storage` — pyre owns the
    /// `ModuleDictStorage` directly (no `rerased` indirection); return
    /// the storage as an erased `*mut u8` so the trait surface stays
    /// strategy-agnostic.
    fn get_empty_storage(&self) -> *mut u8 {
        crate::lltype::malloc_raw(ModuleDictStorage::new()) as *mut u8
    }

    /// `celldict.py:131-141 getitem` — str fast path, else
    /// `switch_to_object_strategy` then walk unified entries.
    /// Body in `w_module_dict_lookup_inner` to avoid recursing
    /// through `w_dict_lookup` (which dispatches back through
    /// the strategy slot after Phase C-3 wire-in).
    unsafe fn getitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> Option<PyObjectRef> {
        crate::dictmultiobject::w_module_dict_lookup_inner(w_dict, w_key)
    }

    /// `celldict.py:96-100 getitem_str` — str fast path matches
    /// `w_module_dict_getitem_str` (which honours `dict_storage_proxy`
    /// and the `mstrategy.caches` cell-cache when present).
    unsafe fn getitem_str(&self, w_dict: PyObjectRef, key: &str) -> Option<PyObjectRef> {
        crate::dictmultiobject::w_module_dict_getitem_str(w_dict, key)
    }

    /// `celldict.py:41-67 setitem` + `_setitem_str_cell_known` — str
    /// fast path; non-str keys force `switch_to_object_strategy`.
    /// Body in `w_module_dict_store_inner` to avoid recursing through
    /// `w_dict_store` (which dispatches back through the strategy
    /// slot after Phase C-3 wire-in).
    unsafe fn setitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef, w_value: PyObjectRef) {
        crate::dictmultiobject::w_module_dict_store_inner(w_dict, w_key, w_value);
    }

    /// `celldict.py:69-74 setitem_str`.
    unsafe fn setitem_str(&self, w_dict: PyObjectRef, key: &str, w_value: PyObjectRef) {
        crate::dictmultiobject::w_module_dict_setitem_str(w_dict, key, w_value);
    }

    /// `celldict.py:106-126 delitem` — str fast path, else
    /// `switch_to_object_strategy` then walk unified entries.
    /// Body in `w_module_dict_delitem_inner` to avoid recursing
    /// through `w_dict_delitem` (which dispatches back through
    /// the strategy slot after Phase C-3 wire-in).
    unsafe fn delitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> bool {
        crate::dictmultiobject::w_module_dict_delitem_inner(w_dict, w_key)
    }

    /// `celldict.py:85-86 length`.
    unsafe fn length(&self, w_dict: PyObjectRef) -> usize {
        crate::dictmultiobject::w_module_dict_length(w_dict)
    }

    /// `celldict.py:140-142 w_keys` — `space.newlist(self.unerase
    /// (w_dict.dstorage).keys())`; pyre returns the wrapped key
    /// PyObjectRefs directly so callers can build whatever container
    /// they need.
    unsafe fn w_keys(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        crate::dictmultiobject::w_dict_items(w_dict)
            .into_iter()
            .map(|(k, _)| k)
            .collect()
    }

    /// `celldict.py:144-149 values`.
    unsafe fn values(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        crate::dictmultiobject::w_dict_items(w_dict)
            .into_iter()
            .map(|(_, v)| v)
            .collect()
    }

    /// `celldict.py:151-155 items` — branches on `is_object_strategy`
    /// and emits whichever storage half is live, wrapping str keys
    /// via `w_str_new`.
    unsafe fn items(&self, w_dict: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
        crate::dictmultiobject::w_module_dict_items_inner(w_dict)
    }

    /// `celldict.py:157-159 clear`.  Branches on
    /// `is_object_strategy` and drains whichever storage half is
    /// live; `w_dict_clear` (the public wrapper) handles
    /// `dict_storage_proxy` flush bookkeeping.
    unsafe fn clear(&self, w_dict: PyObjectRef) {
        crate::dictmultiobject::w_module_dict_clear_inner(w_dict);
    }

    /// `celldict.py:166-173 popitem` — pop the most recently inserted
    /// (key, cell) from the IndexMap, mutated(), unwrap the cell, and
    /// return (`_wrapkey(space, key)`, `unwrap_cell(space, cell)`).
    /// O(1) via `IndexMap::pop`; falls back to the trait default
    /// after a `switch_to_object_strategy` (entries live in
    /// `object_storage`).
    unsafe fn popitem(&self, w_dict: PyObjectRef) -> Option<(PyObjectRef, PyObjectRef)> {
        if let Some(entries) = crate::dictmultiobject::w_module_dict_object_storage_mut_opt(w_dict)
        {
            let (k, v) = entries.pop()?;
            return Some((k.obj, v));
        }
        let module = &mut *(w_dict as *mut crate::dictmultiobject::W_ModuleDictObject);
        let strategy = &mut *module.mstrategy;
        let storage = &mut *module.dstorage;
        let (key, cell) = storage.entries.pop()?;
        strategy.mutated();
        Some((crate::w_str_new(&key), unwrap_cell(cell)))
    }

    /// `celldict.py:198-199 getiterreversed` — reverse iteration
    /// over the IndexMap's key insertion order (used by `reversed
    /// (module.__dict__)`).  Native streaming reverse via
    /// `IndexMap::iter().rev()`; the wrap_cell unwrap matches PyPy's
    /// `wrapvalue(space, value) = unwrap_cell(space, value)` per
    /// `:208 wrapvalue`.
    unsafe fn getiterreversed(&self, w_dict: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
        if let Some(entries) = crate::dictmultiobject::w_module_dict_object_storage(w_dict) {
            return entries.iter().rev().map(|(k, &v)| (k.obj, v)).collect();
        }
        let module = &*(w_dict as *const crate::dictmultiobject::W_ModuleDictObject);
        let storage = &*module.dstorage;
        storage
            .entries
            .iter()
            .rev()
            .map(|(k, &cell)| (crate::w_str_new(k), unwrap_cell(cell)))
            .collect()
    }

    /// `celldict.py:207-216 copy` — produce a fresh W_DictObject that
    /// owns unwrapped cell values keyed by str objects.
    ///
    /// PyPy's destination strategy is `UnicodeDictStrategy`; pyre
    /// still allocates the destination through `w_dict_new()` (which
    /// installs `OBJECT_DICT_STRATEGY`) — line-by-line parity for the
    /// destination strategy lands when typed strategy storage is in
    /// place.
    unsafe fn copy(&self, w_dict: PyObjectRef) -> PyObjectRef {
        let new_dict = crate::dictmultiobject::w_dict_new();
        if let Some(entries) = crate::dictmultiobject::w_module_dict_object_storage(w_dict) {
            for (k, &v) in entries.iter() {
                crate::dictmultiobject::w_dict_store(new_dict, k.obj, v);
            }
            return new_dict;
        }
        let module = &*(w_dict as *const crate::dictmultiobject::W_ModuleDictObject);
        let storage = &*module.dstorage;
        for (key, &cell) in storage.entries.iter() {
            let unwrapped = unwrap_cell(cell);
            let key_obj = crate::w_str_new(key);
            crate::dictmultiobject::w_dict_store(new_dict, key_obj, unwrapped);
        }
        new_dict
    }
}

/// `celldict.py:243-251 remove_cell(w_dict, space, name)` — replace
/// any cell wrapper at `name` with the unwrapped value so subsequent
/// reads observe the raw value directly (used when a module-level
/// name is rebound in a context that no longer needs cell
/// indirection, e.g. function-def replacing a previously
/// cell-promoted slot).
///
/// ```python
/// def remove_cell(w_dict, space, name):
///     if isinstance(w_dict, W_DictMultiObject):
///         strategy = w_dict.get_strategy()
///         if isinstance(strategy, ModuleDictStrategy):
///             w_value = strategy.getitem_str(w_dict, name)
///             dict_w = strategy.unerase(w_dict.dstorage)
///             strategy.mutated()
///             dict_w[name] = w_value  # store without cell
/// ```
///
/// Pyre's W_ModuleDictObject path: peek the unwrapped value via
/// `getitem_str` (which already calls `unwrap_cell`), bump the
/// strategy version (cache invalidate), and write back the raw
/// PyObjectRef via `ModuleDictStorage::set` — bypassing
/// `_setitem_str_cell_known`'s `write_cell` re-wrap.
///
/// # Safety
/// `w_dict` must point at a valid PyObjectRef (W_ModuleDictObject
/// or null/other type — no-op for non-module dicts).
pub unsafe fn remove_cell(w_dict: PyObjectRef, name: &str) {
    if w_dict.is_null() {
        return;
    }
    if (*(w_dict as *const crate::pyobject::PyObject)).ob_type
        != &crate::dictmultiobject::MODULE_DICT_TYPE as *const crate::pyobject::PyType
    {
        return;
    }
    let module = &mut *(w_dict as *mut crate::dictmultiobject::W_ModuleDictObject);
    let strategy = &mut *module.mstrategy;
    let storage = &mut *module.dstorage;
    let Some(w_value) = strategy.getitem_str(storage, name) else {
        return;
    };
    strategy.mutated();
    storage.set(name, w_value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_versiontags_are_distinct() {
        let a = VersionTag::fresh();
        let b = VersionTag::fresh();
        assert_ne!(a, b);
    }

    #[test]
    fn mutated_flips_registered_version_watchers() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        let mut strategy = ModuleDictStrategy::new();
        let flag = Arc::new(AtomicBool::new(false));
        strategy.register_version_watcher(&flag);
        // Before a structural change the watching loop is still valid.
        assert_eq!(flag.load(Ordering::Acquire), false);
        // `mutated()` reassigns `version` and must invalidate the loop.
        strategy.mutated();
        assert!(flag.load(Ordering::Acquire));
    }

    #[test]
    fn dead_version_watchers_drop_out() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        let mut strategy = ModuleDictStrategy::new();
        let flag = Arc::new(AtomicBool::new(false));
        strategy.register_version_watcher(&flag);
        // Drop the only strong ref: the weak watcher can no longer upgrade.
        drop(flag);
        // notify (via mutated) must not panic and must purge the dead weak.
        strategy.mutated();
        assert!(strategy.version_watchers.is_empty());
    }

    #[test]
    fn setitem_getitem_roundtrip() {
        let mut strat = ModuleDictStrategy::new();
        let mut store = strat.get_empty_storage();
        let v = crate::w_str_new("hello");
        strat.setitem_str(&mut store, "x", v);
        assert_eq!(strat.getitem_str(&store, "x"), Some(v));
        assert_eq!(strat.length(&store), 1);
    }

    #[test]
    fn setitem_bumps_version() {
        let mut strat = ModuleDictStrategy::new();
        let before = strat.version;
        let mut store = strat.get_empty_storage();
        strat.setitem_str(&mut store, "k", crate::w_str_new("v"));
        assert_ne!(strat.version, before);
    }

    #[test]
    fn delitem_removes_and_bumps() {
        let mut strat = ModuleDictStrategy::new();
        let mut store = strat.get_empty_storage();
        let v = crate::w_str_new("v");
        strat.setitem_str(&mut store, "k", v);
        let v_before = strat.version;
        let removed = strat.delitem_str(&mut store, "k");
        assert_eq!(removed, Some(v));
        assert_eq!(strat.getitem_str(&store, "k"), None);
        assert_ne!(strat.version, v_before);
    }

    #[test]
    fn int_cell_indirection_on_rewrite() {
        // After a second write with an int value, the strategy should
        // wrap the value in `IntMutableCell` and skip the version bump
        // (typeobject.py:61-63).
        let mut strat = ModuleDictStrategy::new();
        let mut store = strat.get_empty_storage();
        unsafe {
            let v0 = crate::intobject::w_int_new(7);
            strat.setitem_str(&mut store, "k", v0);
            let v1 = crate::intobject::w_int_new(8);
            strat.setitem_str(&mut store, "k", v1);
            // getitem_str unwraps the cell back to the int value.
            let got = strat.getitem_str(&store, "k").unwrap();
            assert_eq!(crate::intobject::w_int_get_value(got), 8);
        }
    }
}
