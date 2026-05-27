//! `pypy/objspace/std/dictmultiobject.py:449-470 W_DictMultiViewKeysObject`
//! / `W_DictMultiViewValuesObject` / `W_DictMultiViewItemsObject`
//! parity port.
//!
//! PyPy keeps three sibling W_Root types — one per view kind — that
//! all share the same shape: a back-reference to the source
//! `W_DictMultiObject` plus the iteration discipline appropriate to
//! the kind.  Pyre fuses them into a single `W_DictView` carrying a
//! `DictViewKind` tag so the three Python-visible types can share
//! the GC-traced `w_dict` slot and accessors; type identity is
//! restored at the W_TypeObject layer through the kind tag (see
//! `dict_view_type_for_kind`).

use crate::pyobject::*;

pub static DICT_KEYS_TYPE: PyType = crate::pyobject::new_pytype("dict_keys");
pub static DICT_VALUES_TYPE: PyType = crate::pyobject::new_pytype("dict_values");
pub static DICT_ITEMS_TYPE: PyType = crate::pyobject::new_pytype("dict_items");

/// `dictmultiobject.py:449/459/469` — three sibling view classes.
/// Pyre folds them into one struct + tag because the body is
/// otherwise identical (only the iteration / repr shape differs).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DictViewKind {
    Keys = 0,
    Values = 1,
    Items = 2,
}

/// Layout: `[ob_header | kind: DictViewKind | w_dict: PyObjectRef]`.
///
/// `w_dict` is the live `W_DictObject` the view is attached to; PyPy's
/// `W_DictMultiViewKeysObject.w_dict` (`dictmultiobject.py:451`) plays
/// the same role.  Mutations on the source dict are visible through
/// the view because every reader (iter / len / contains) goes through
/// `w_dict` rather than caching a snapshot.
#[repr(C)]
pub struct W_DictView {
    pub ob_header: PyObject,
    pub kind: DictViewKind,
    pub w_dict: PyObjectRef,
}

pub const DICT_VIEW_KIND_OFFSET: usize = std::mem::offset_of!(W_DictView, kind);
pub const DICT_VIEW_W_DICT_OFFSET: usize = std::mem::offset_of!(W_DictView, w_dict);

/// GC type id assigned to `W_DictView` at JitDriver init time.
/// 32 is taken by `W_GENERATOR_GC_TYPE_ID`; the next free slot is 39
/// (one past `W_DICT_PROXY_GC_TYPE_ID = 38`).
pub const W_DICT_VIEW_GC_TYPE_ID: u32 = 39;

pub const W_DICT_VIEW_OBJECT_SIZE: usize = std::mem::size_of::<W_DictView>();

/// Single inline `PyObjectRef`-shaped field — the back-pointer to the
/// source dict.
pub const W_DICT_VIEW_GC_PTR_OFFSETS: [usize; 1] = [DICT_VIEW_W_DICT_OFFSET];

impl crate::lltype::GcType for W_DictView {
    fn type_id() -> u32 {
        W_DICT_VIEW_GC_TYPE_ID
    }
    const SIZE: usize = W_DICT_VIEW_OBJECT_SIZE;
}

/// Pick the Python-visible type for a given view kind.  Mirrors
/// PyPy's three distinct W_TypeObject identities so
/// `type(d.keys()) is dict_keys`, `type(d.values()) is dict_values`,
/// `type(d.items()) is dict_items` all hold.
pub fn dict_view_type_for_kind(kind: DictViewKind) -> &'static PyType {
    match kind {
        DictViewKind::Keys => &DICT_KEYS_TYPE,
        DictViewKind::Values => &DICT_VALUES_TYPE,
        DictViewKind::Items => &DICT_ITEMS_TYPE,
    }
}

/// Allocate a fresh dict view bound to `w_dict`.
pub fn w_dict_view_new(w_dict: PyObjectRef, kind: DictViewKind) -> PyObjectRef {
    let tp = dict_view_type_for_kind(kind);
    crate::lltype::malloc_typed(W_DictView {
        ob_header: PyObject {
            ob_type: tp as *const PyType,
            w_class: get_instantiate(tp),
        },
        kind,
        w_dict,
    }) as PyObjectRef
}

/// Test whether `obj` is any of the three view types.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_dict_view(obj: PyObjectRef) -> bool {
    unsafe {
        py_type_check(obj, &DICT_KEYS_TYPE)
            || py_type_check(obj, &DICT_VALUES_TYPE)
            || py_type_check(obj, &DICT_ITEMS_TYPE)
    }
}

/// # Safety
/// `obj` must point to a valid `W_DictView`.
#[inline]
pub unsafe fn w_dict_view_get_kind(obj: PyObjectRef) -> DictViewKind {
    unsafe { (*(obj as *const W_DictView)).kind }
}

/// # Safety
/// `obj` must point to a valid `W_DictView`.
#[inline]
pub unsafe fn w_dict_view_get_dict(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_DictView)).w_dict }
}

// ── W_DictViewIterator ──
//
// `pypy/objspace/std/dictmultiobject.py:1701-1741 W_BaseDictIterator`
// (and the three concrete kind subclasses `W_DictMultiIterKeys` /
// `W_DictMultiIterValues` / `W_DictMultiIterItems`) line-by-line
// port.  PyPy's iterator captures the source dict + a strategy-specific
// iterator into the entries; mutation tracking happens via the
// `len(w_dict) != self.startlen` check inside `next_w` per
// `:1727-1733`, which raises `RuntimeError("dictionary changed size
// during iteration")`.
//
// Pyre's flat entries Vec lets us index directly; the parity-correct
// detection compares `dict.version` against the version captured at
// iter() time, matching PyPy's `dictversion` check in
// `:1701-1741 W_BaseDictIterator`.

pub static DICT_KEYITERATOR_TYPE: PyType = crate::pyobject::new_pytype("dict_keyiterator");
pub static DICT_VALUEITERATOR_TYPE: PyType = crate::pyobject::new_pytype("dict_valueiterator");
pub static DICT_ITEMITERATOR_TYPE: PyType = crate::pyobject::new_pytype("dict_itemiterator");

/// `dictmultiobject.py:809-845 _new_next` — captures both the source
/// dict's `len` and the active strategy at iter() time; `next()`
/// performs two parity-mandated checks per `:821-822` and `:829`:
///   * `self.len != self.w_dict.length()` → RuntimeError
///     ("dictionary changed size during iteration").  Re-assigning
///     an existing key inside a loop (`for k in d: d[k] = new`) is
///     permitted because `length()` is unchanged.
///   * `self.strategy is not self.w_dict.get_strategy()` → strategy
///     transition (e.g. `switch_to_object_strategy` mid-iteration).
///     The (key, value) pair in `result` may be out-of-date; PyPy
///     re-looks up the key in the dict and raises "dictionary
///     changed during iteration" if the key was removed (`:837-841`).
///     Keys/values iterators accept the stale result.
#[repr(C)]
pub struct W_DictViewIterator {
    pub ob_header: PyObject,
    /// `:1707 self.w_dict` — back-pointer to the live source dict.
    pub w_dict: PyObjectRef,
    /// `:809 self.len = w_dict.length()` parity — captures len at
    /// iter() time; iterator compares against `w_dict.length()` on
    /// each `next()` step.
    pub startlen: usize,
    /// Iteration cursor into the source dict's entries Vec.
    pub index: usize,
    /// `DictViewKind` repurposed for the three concrete iterator
    /// kinds (`W_DictMultiIterKeys` / `Values` / `Items` —
    /// `:1499-1538`).
    pub kind: DictViewKind,
    /// `:807 self.strategy = strategy` — strategy identity at iter()
    /// time, stored as the strategy pointer cast to `usize` for
    /// identity comparison (PyPy's `self.strategy is
    /// self.w_dict.get_strategy()` at `:829`).  Strategy singletons
    /// are `'static` so the cast is stable across the iterator's
    /// lifetime.
    pub start_strategy_id: usize,
}

pub const DICT_VIEW_ITER_W_DICT_OFFSET: usize = std::mem::offset_of!(W_DictViewIterator, w_dict);

/// GC type id — next free slot after enumerate (=41).
pub const W_DICT_VIEW_ITERATOR_GC_TYPE_ID: u32 = 42;
pub const W_DICT_VIEW_ITERATOR_OBJECT_SIZE: usize = std::mem::size_of::<W_DictViewIterator>();

pub const W_DICT_VIEW_ITERATOR_GC_PTR_OFFSETS: [usize; 1] = [DICT_VIEW_ITER_W_DICT_OFFSET];

impl crate::lltype::GcType for W_DictViewIterator {
    fn type_id() -> u32 {
        W_DICT_VIEW_ITERATOR_GC_TYPE_ID
    }
    const SIZE: usize = W_DICT_VIEW_ITERATOR_OBJECT_SIZE;
}

/// Pick the Python-visible iterator type for a given view kind so
/// `type(iter(d.keys())) is dict_keyiterator` (etc.).
pub fn dict_view_iterator_type_for_kind(kind: DictViewKind) -> &'static PyType {
    match kind {
        DictViewKind::Keys => &DICT_KEYITERATOR_TYPE,
        DictViewKind::Values => &DICT_VALUEITERATOR_TYPE,
        DictViewKind::Items => &DICT_ITEMITERATOR_TYPE,
    }
}

/// Allocate a fresh dict iterator capturing `w_dict`'s current length
/// and active strategy.  Mirrors `dictmultiobject.py:807-808
/// W_BaseIteratorImplementation.__init__`:
///
/// ```python
/// self.strategy = strategy
/// self.len = w_dict.length()
/// ```
pub fn w_dict_view_iterator_new(w_dict: PyObjectRef, kind: DictViewKind) -> PyObjectRef {
    let startlen = unsafe { crate::dictmultiobject::w_dict_len(w_dict) };
    let start_strategy_id = unsafe { crate::dictmultiobject::w_dict_strategy_id(w_dict) };
    let tp = dict_view_iterator_type_for_kind(kind);
    crate::lltype::malloc_typed(W_DictViewIterator {
        ob_header: PyObject {
            ob_type: tp as *const PyType,
            w_class: get_instantiate(tp),
        },
        w_dict,
        startlen,
        index: 0,
        kind,
        start_strategy_id,
    }) as PyObjectRef
}

/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_dict_view_iterator(obj: PyObjectRef) -> bool {
    unsafe {
        py_type_check(obj, &DICT_KEYITERATOR_TYPE)
            || py_type_check(obj, &DICT_VALUEITERATOR_TYPE)
            || py_type_check(obj, &DICT_ITEMITERATOR_TYPE)
    }
}

/// # Safety
/// `obj` must point to a valid `W_DictViewIterator`.
#[inline]
pub unsafe fn w_dict_view_iterator_get_dict(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_DictViewIterator)).w_dict }
}

/// # Safety
/// `obj` must point to a valid `W_DictViewIterator`.
#[inline]
pub unsafe fn w_dict_view_iterator_get_kind(obj: PyObjectRef) -> DictViewKind {
    unsafe { (*(obj as *const W_DictViewIterator)).kind }
}

/// # Safety
/// `obj` must point to a valid `W_DictViewIterator`.
#[inline]
pub unsafe fn w_dict_view_iterator_get_startlen(obj: PyObjectRef) -> usize {
    unsafe { (*(obj as *const W_DictViewIterator)).startlen }
}

/// # Safety
/// `obj` must point to a valid `W_DictViewIterator`.
#[inline]
pub unsafe fn w_dict_view_iterator_get_index(obj: PyObjectRef) -> usize {
    unsafe { (*(obj as *const W_DictViewIterator)).index }
}

/// # Safety
/// `obj` must point to a valid `W_DictViewIterator`.
#[inline]
pub unsafe fn w_dict_view_iterator_set_index(obj: PyObjectRef, value: usize) {
    unsafe {
        (*(obj as *mut W_DictViewIterator)).index = value;
    }
}

/// `:807 self.strategy = strategy` — strategy id captured at iter()
/// creation.  Consumers compare it against the dict's current
/// `w_dict_strategy_id` to detect strategy transitions
/// (`dictmultiobject.py:829`).
///
/// # Safety
/// `obj` must point to a valid `W_DictViewIterator`.
#[inline]
pub unsafe fn w_dict_view_iterator_get_start_strategy_id(obj: PyObjectRef) -> usize {
    unsafe { (*(obj as *const W_DictViewIterator)).start_strategy_id }
}
