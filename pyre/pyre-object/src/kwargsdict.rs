//! `pypy/objspace/std/kwargsdict.py` port — dict implementation
//! specialized for keyword argument dicts.
//!
//! Based on two parallel lists `(keys_w, values_w)` of `PyObjectRef`.
//! Optimized for the common `**kwargs` shape: a small number of
//! distinct string keys with O(n) linear-scan lookup that the JIT
//! constant-folds when the dict size and lookup key are both
//! constant.
//!
//! `EmptyKwargsDictStrategy` (`kwargsdict.py:13-22`) — the subclass
//! of `EmptyDictStrategy` that promotes to `KwargsDictStrategy` on
//! unicode setitem — is not yet wired into pyre's argument
//! processing path; once it lands, function-call sites can
//! allocate dicts via the subclassed empty strategy.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::dictmultiobject::{DictStrategy, OBJECT_DICT_STRATEGY};
use crate::pyobject::PyObjectRef;

/// `kwargsdict.py:25-178 KwargsDictStrategy`.
///
/// ```python
/// class KwargsDictStrategy(DictStrategy):
///     erase, unerase = rerased.new_erasing_pair("kwargsdict")
///
///     def get_empty_storage(self):
///         d = ([], [])
///         return self.erase(d)
///
///     def is_correct_type(self, w_obj):
///         space = self.space
///         return space.is_w(space.type(w_obj), space.w_text)
///
///     def setitem(self, w_dict, w_key, w_value):
///         if self.is_correct_type(w_key):
///             self.setitem_correct(w_dict, w_key, w_value)
///             return
///         else:
///             self.switch_to_object_strategy(w_dict)
///             w_dict.setitem(w_key, w_value)
/// ```
///
/// Two-list backing chosen because:
/// - Function-call sites always create small kwarg dicts.
/// - The JIT can fold the entire lookup loop when both size and key
///   are constant via `jit.look_inside_iff`.
/// - At size ≥ 16 entries (`kwargsdict.py:62`) the strategy
///   auto-promotes to `UnicodeDictStrategy` to avoid degenerate O(n).
pub struct KwargsDictStrategy;

/// `pypy/objspace/std/kwargsdict.py:25 KwargsDictStrategy`
/// singleton — matches PyPy's `space.fromcache(KwargsDictStrategy)`.
pub static KWARGS_DICT_STRATEGY: KwargsDictStrategy = KwargsDictStrategy;

/// `KwargsDictStrategy` backing — erased `([], [])` parallel arrays
/// (`kwargsdict.py:27-29`). GC-managed storage box (mirrors the other
/// dict strategies; see `dictmultiobject::ObjectDictStorage`).
pub type KwargsDictStorage = (Vec<PyObjectRef>, Vec<PyObjectRef>);

/// Runtime-assigned GC type id for the [`KwargsDictStorage`] box.
static KWARGS_DICT_STORAGE_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for the [`KwargsDictStorage`] box.
pub fn set_kwargs_dict_storage_gc_type_id(id: u32) {
    KWARGS_DICT_STORAGE_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for the [`KwargsDictStorage`] box.
#[majit_macros::dont_look_inside]
pub fn kwargs_dict_storage_gc_type_id() -> u32 {
    KWARGS_DICT_STORAGE_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// `kwargsdict.py:62` size threshold past which the strategy
/// promotes itself to UnicodeDictStrategy to avoid O(n) lookup
/// degeneracy on too-large kwarg dicts.
const KWARGS_PROMOTE_THRESHOLD: usize = 16;

/// Typed accessor for `KwargsDictStrategy.unerase(w_dict.dstorage)` —
/// `kwargsdict.py:26-32` parallel-array shape.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject` whose strategy is
/// [`KWARGS_DICT_STRATEGY`].
#[inline]
unsafe fn kwargs_storage<'a>(obj: PyObjectRef) -> &'a (Vec<PyObjectRef>, Vec<PyObjectRef>) {
    let dict = &*(obj as *const crate::dictmultiobject::W_DictObject);
    &*(dict.dstorage as *const (Vec<PyObjectRef>, Vec<PyObjectRef>))
}

#[inline]
unsafe fn kwargs_storage_mut<'a>(obj: PyObjectRef) -> &'a mut (Vec<PyObjectRef>, Vec<PyObjectRef>) {
    let dict = &mut *(obj as *mut crate::dictmultiobject::W_DictObject);
    &mut *(dict.dstorage as *mut (Vec<PyObjectRef>, Vec<PyObjectRef>))
}

impl KwargsDictStrategy {
    /// `kwargsdict.py:34-36 is_correct_type` — `space.is_w
    /// (space.type(w_obj), space.w_text)`.  Plain str (Py3 unicode).
    #[inline]
    unsafe fn is_correct_type(w_key: PyObjectRef) -> bool {
        crate::is_exact_type(w_key, &crate::STR_TYPE)
    }

    /// `kwargsdict.py:143-152 switch_to_unicode_strategy` —
    /// promote to UnicodeDictStrategy when size hits the threshold.
    /// PyPy walks the parallel arrays and re-inserts each entry via
    /// `w_dict.setitem`; pyre does the same so any non-ASCII keys
    /// further promote to ObjectDictStrategy.
    unsafe fn switch_to_unicode_strategy(&self, w_dict: PyObjectRef) {
        let dict = &mut *(w_dict as *mut crate::dictmultiobject::W_DictObject);
        // Drain the parallel arrays out of the old box (leaving it holding
        // empty Vecs); after `dstorage` is overwritten the box is unreachable
        // and the sweep drops it. `std::mem::take` mirrors the old
        // `Box::from_raw` move without freeing the GC-managed box here.
        let old = &mut *(dict.dstorage as *mut KwargsDictStorage);
        let keys_w = std::mem::take(&mut old.0);
        let values_w = std::mem::take(&mut old.1);
        dict.dstorage = crate::dictmultiobject::UNICODE_DICT_STRATEGY.get_empty_storage();
        dict.dstrategy = &crate::dictmultiobject::UNICODE_DICT_STRATEGY;
        for (k, v) in keys_w.into_iter().zip(values_w.into_iter()) {
            crate::dictmultiobject::w_dict_store(w_dict, k, v);
        }
    }
}

impl DictStrategy for KwargsDictStrategy {
    fn strategy_kind(&self) -> crate::dictmultiobject::StrategyKind {
        crate::dictmultiobject::StrategyKind::Kwargs
    }

    /// `kwargsdict.py:30-32 get_empty_storage` — erased `([], [])`.
    /// GC-managed box (`setfield_gc` on reassign).
    fn get_empty_storage(&self) -> *mut u8 {
        crate::gc_storage::gc_alloc_storage_box(
            KwargsDictStorage::default(),
            kwargs_dict_storage_gc_type_id(),
        ) as *mut u8
    }

    /// `kwargsdict.py:134-141 switch_to_object_strategy` — walk
    /// parallel arrays, rebuild `IndexMap<ObjectKey, PyObjectRef>`,
    /// retire the typed parallel-array box.
    unsafe fn switch_to_object_strategy(&self, w_dict: PyObjectRef) {
        let dict = &mut *(w_dict as *mut crate::dictmultiobject::W_DictObject);
        // Borrow the old typed box (its field stays live, so it is traced
        // while the migration builds the object map); after the store the
        // box is unreachable and the sweep reclaims it. `PyObjectRef` is a
        // raw pointer (Copy), so iterate by reference and copy each slot.
        let old = &*(dict.dstorage as *const KwargsDictStorage);
        let mut new_map = crate::dictmultiobject::ObjectDictStorage::with_capacity(old.0.len());
        for (k, v) in old.0.iter().zip(old.1.iter()) {
            new_map.insert(crate::dictmultiobject::object_key_for(*k), *v);
        }
        dict.dstorage = crate::gc_storage::gc_alloc_storage_box(
            new_map,
            crate::dictmultiobject::object_dict_storage_gc_type_id(),
        ) as *mut u8;
        dict.dstrategy = &OBJECT_DICT_STRATEGY;
    }

    /// `kwargsdict.py:100-108 getitem` — `is_correct_type` →
    /// linear scan, else `_never_equal_to` short-circuit or promote.
    unsafe fn getitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> Option<PyObjectRef> {
        if Self::is_correct_type(w_key) {
            let (keys_w, values_w) = kwargs_storage(w_dict);
            for i in 0..keys_w.len() {
                if crate::dictmultiobject::dict_keys_equal(keys_w[i], w_key) {
                    return Some(values_w[i]);
                }
            }
            return None;
        }
        // `kwargsdict.py:38-39 _never_equal_to` returns False — no
        // short-circuit; always promote and retry.
        self.switch_to_object_strategy(w_dict);
        crate::dictmultiobject::w_dict_lookup(w_dict, w_key)
    }

    /// `kwargsdict.py:41-67 setitem` + `_setitem_correct_indirection`.
    unsafe fn setitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef, w_value: PyObjectRef) {
        if Self::is_correct_type(w_key) {
            let dict = &mut *(w_dict as *mut crate::dictmultiobject::W_DictObject);
            let storage = &mut *(dict.dstorage as *mut (Vec<PyObjectRef>, Vec<PyObjectRef>));
            for i in 0..storage.0.len() {
                if crate::dictmultiobject::dict_keys_equal(storage.0[i], w_key) {
                    storage.1[i] = w_value;
                    crate::gc_hook::try_gc_write_barrier(w_dict as *mut u8);
                    return;
                }
            }
            if storage.0.len() >= KWARGS_PROMOTE_THRESHOLD {
                self.switch_to_unicode_strategy(w_dict);
                crate::dictmultiobject::w_dict_store(w_dict, w_key, w_value);
                return;
            }
            storage.0.push(w_key);
            storage.1.push(w_value);
            crate::dictmultiobject::w_dict_bump_keys_version(w_dict);
            crate::gc_hook::try_gc_write_barrier(w_dict as *mut u8);
            return;
        }
        self.switch_to_object_strategy(w_dict);
        crate::dictmultiobject::w_dict_store(w_dict, w_key, w_value);
    }

    /// `kwargsdict.py:80-83 delitem` — switches to object strategy
    /// first (XXX comment: "could do better but is it worth it?").
    unsafe fn delitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> bool {
        self.switch_to_object_strategy(w_dict);
        crate::dictmultiobject::w_dict_delitem(w_dict, w_key)
    }

    /// `kwargsdict.py:85-86 length`.
    unsafe fn length(&self, w_dict: PyObjectRef) -> usize {
        kwargs_storage(w_dict).0.len()
    }

    /// `kwargsdict.py:110-112 w_keys` — returns a copy of `keys_w`.
    unsafe fn w_keys(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        kwargs_storage(w_dict).0.clone()
    }

    /// `kwargsdict.py:114-115 values`.
    unsafe fn values(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        kwargs_storage(w_dict).1.clone()
    }

    /// `kwargsdict.py:117-121 items` — zip parallel arrays into pairs.
    unsafe fn items(&self, w_dict: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
        let (keys_w, values_w) = kwargs_storage(w_dict);
        keys_w
            .iter()
            .zip(values_w.iter())
            .map(|(&k, &v)| (k, v))
            .collect()
    }

    /// `kwargsdict.py:123-129 popitem` — pop from both arrays in lock-step.
    unsafe fn popitem(&self, w_dict: PyObjectRef) -> Option<(PyObjectRef, PyObjectRef)> {
        let storage = kwargs_storage_mut(w_dict);
        let w_key = storage.0.pop()?;
        let w_value = storage.1.pop()?;
        crate::dictmultiobject::w_dict_bump_keys_version(w_dict);
        Some((w_key, w_value))
    }

    /// `kwargsdict.py:131-132 clear` — `w_dict.dstorage =
    /// self.get_empty_storage()`.  The field overwrite is a `setfield_gc`;
    /// the unreachable old parallel-array box is reclaimed by the sweep.
    unsafe fn clear(&self, w_dict: PyObjectRef) {
        let dict = &mut *(w_dict as *mut crate::dictmultiobject::W_DictObject);
        let storage = &*(dict.dstorage as *const (Vec<PyObjectRef>, Vec<PyObjectRef>));
        if !storage.0.is_empty() {
            dict.keys_version = dict.keys_version.wrapping_add(1);
        }
        dict.dstorage = self.get_empty_storage();
    }

    /// `kwargsdict.py:154-156 view_as_kwargs` — copy parallel arrays
    /// to non-resizable slices for the `**kwargs` fast unpack.
    unsafe fn view_as_kwargs(
        &self,
        w_dict: PyObjectRef,
    ) -> (Option<Vec<PyObjectRef>>, Option<Vec<PyObjectRef>>) {
        let (keys_w, values_w) = kwargs_storage(w_dict);
        (Some(keys_w.clone()), Some(values_w.clone()))
    }

    /// `kwargsdict.py` traces both `keys_w` and `values_w` as
    /// `list[W_Root]` — every entry on both sides is PyObjectRef.
    unsafe fn walk_gc_refs(&self, w_dict: PyObjectRef, visitor: &mut dyn FnMut(*mut PyObjectRef)) {
        let storage = kwargs_storage_mut(w_dict);
        for k in storage.0.iter_mut() {
            visitor(k as *mut PyObjectRef);
        }
        for v in storage.1.iter_mut() {
            visitor(v as *mut PyObjectRef);
        }
    }

    /// `dictmultiobject.py:1152 AbstractTypedStrategy.copy` — clone
    /// the parallel `(keys_w, values_w)` arrays and wrap with the
    /// same KwargsDictStrategy.
    unsafe fn copy(&self, w_dict: PyObjectRef) -> PyObjectRef {
        let storage = kwargs_storage(w_dict);
        let new_storage = crate::gc_storage::gc_alloc_storage_box(
            storage.clone(),
            kwargs_dict_storage_gc_type_id(),
        );
        crate::dictmultiobject::w_dict_new_with(&KWARGS_DICT_STRATEGY, new_storage as *mut u8)
    }
}
