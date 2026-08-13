//! `pypy/objspace/std/kwargsdict.py` port — dict implementation
//! specialized for keyword argument dicts.
//!
//! Based on two parallel lists `(keys_w, values_w)` of `PyObjectRef`.
//! Optimized for the common `**kwargs` shape: a small number of
//! distinct string keys with O(n) linear-scan lookup that the JIT
//! constant-folds when the dict size and lookup key are both
//! constant.
//!
//! `EmptyKwargsDictStrategy` (`kwargsdict.py:13-22`) is selected by
//! `w_dict_new_kwargs`; function-call `**kwargs` collectors allocate through
//! that entry point and the first unicode store promotes directly to this
//! parallel-array strategy.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::dictmultiobject::DictStrategy;
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

/// The [`crate::dictmultiobject::DictStrategyRef`] holder a dict's `dstrategy`
/// slot points at.
pub static KWARGS_DICT_STRATEGY_REF: crate::dictmultiobject::DictStrategyRef =
    crate::dictmultiobject::DictStrategyRef {
        imp: &KWARGS_DICT_STRATEGY,
    };

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

/// `kwargsdict.py:134-141 switch_to_object_strategy` — walk the parallel
/// arrays, rebuild `IndexMap<ObjectKey, PyObjectRef>`, retire the typed
/// parallel-array box.
///
/// Residualised (`@dont_look_inside`, `rlib/jit.py:139`) for the reason
/// `dictmultiobject::w_dict_switch_int_to_object_strategy` is: upstream traces
/// the same loop because every step is an RPython dict primitive the JIT
/// models, whereas this body is `IndexMap` end to end and the front end has no
/// lowering for it, so the last modellable point is the call itself.
///
/// # Safety
/// `w_dict` must be a valid `W_DictObject` on [`KWARGS_DICT_STRATEGY`].
#[majit_macros::dont_look_inside]
pub unsafe fn w_dict_switch_kwargs_to_object_strategy(w_dict: PyObjectRef) {
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
    dict.dstrategy = &crate::dictmultiobject::OBJECT_DICT_STRATEGY_REF;
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
        dict.dstrategy = &crate::dictmultiobject::UNICODE_DICT_STRATEGY_REF;
        for (k, v) in keys_w.into_iter().zip(values_w) {
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
        w_dict_switch_kwargs_to_object_strategy(w_dict);
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

    /// `kwargsdict.py:68-79 setdefault` — keep an exact unicode key on the
    /// parallel-array strategy, otherwise switch first and re-dispatch on the
    /// new strategy.  The latter detail matters structurally: after the swap
    /// PyPy calls `w_dict.setdefault`, it does not keep invoking methods on
    /// the stale `KwargsDictStrategy` instance.
    unsafe fn setdefault(
        &self,
        w_dict: PyObjectRef,
        w_key: PyObjectRef,
        w_default: PyObjectRef,
    ) -> PyObjectRef {
        if Self::is_correct_type(w_key) {
            if let Some(w_result) = self.getitem(w_dict, w_key) {
                return w_result;
            }
            self.setitem(w_dict, w_key, w_default);
            return w_default;
        }
        self.switch_to_object_strategy(w_dict);
        crate::dictmultiobject::w_dict_get_strategy(w_dict).setdefault(w_dict, w_key, w_default)
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

    /// `create_iterator_classes(KwargsDictStrategy)` reads the two backing
    /// lists at the same cursor position.  Supplying the cursor operation
    /// directly avoids the trait fallback's `items().into_iter().nth(index)`,
    /// which rebuilt the complete kwargs list at every iterator step.
    unsafe fn nth_item(
        &self,
        w_dict: PyObjectRef,
        index: usize,
    ) -> Option<(PyObjectRef, PyObjectRef)> {
        let (keys_w, values_w) = kwargs_storage(w_dict);
        keys_w.get(index).copied().zip(values_w.get(index).copied())
    }

    /// Value-iterator twin of [`Self::nth_item`], matching
    /// `kwargsdict.py:114-115 itervalues`' direct values-list cursor.
    unsafe fn nth_value(&self, w_dict: PyObjectRef, index: usize) -> Option<PyObjectRef> {
        kwargs_storage(w_dict).1.get(index).copied()
    }

    /// `kwargsdict.py:123-129 popitem` — pop from both arrays in lock-step.
    unsafe fn popitem(&self, w_dict: PyObjectRef) -> Option<(PyObjectRef, PyObjectRef)> {
        let storage = kwargs_storage_mut(w_dict);
        let w_key = storage.0.pop()?;
        let w_value = storage.1.pop()?;
        crate::dictmultiobject::w_dict_bump_keys_version(w_dict);
        Some((w_key, w_value))
    }

    /// `kwargsdict.py:178-181 getiterreversed` — copy/reverse the key list in
    /// PyPy.  Pyre's iterator carrier consumes key/value pairs, so walk both
    /// parallel lists from the tail without first materialising `items()`.
    unsafe fn getiterreversed(&self, w_dict: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
        let (keys_w, values_w) = kwargs_storage(w_dict);
        keys_w
            .iter()
            .copied()
            .zip(values_w.iter().copied())
            .rev()
            .collect()
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
        crate::dictmultiobject::w_dict_new_with(&KWARGS_DICT_STRATEGY_REF, new_storage as *mut u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn install_test_hash_hooks() {
        unsafe fn hash_object(obj: PyObjectRef) -> i64 {
            if crate::is_int(obj) {
                crate::w_int_get_value(obj)
            } else {
                0
            }
        }

        unsafe fn hash_str(_ptr: *const u8, _len: usize) -> i64 {
            0
        }

        crate::dict_eq_hook::register_hash_w_hook(hash_object);
        crate::dict_eq_hook::register_hash_str_hook(hash_str);
    }

    unsafe fn kwargs_with(entries: &[(&str, i64)]) -> PyObjectRef {
        let w_dict = crate::dictmultiobject::w_dict_new_kwargs();
        for &(key, value) in entries {
            crate::dictmultiobject::w_dict_setitem_str(w_dict, key, crate::w_int_new(value));
        }
        assert_eq!(
            crate::dictmultiobject::w_dict_get_strategy(w_dict).strategy_kind(),
            crate::dictmultiobject::StrategyKind::Kwargs
        );
        w_dict
    }

    #[test]
    fn kwargs_cursor_reads_parallel_arrays_without_materialising_items() {
        unsafe {
            let w_dict = kwargs_with(&[("a", 10), ("b", 20), ("c", 30)]);
            let strategy = crate::dictmultiobject::w_dict_get_strategy(w_dict);

            for (index, (key, value)) in [("a", 10), ("b", 20), ("c", 30)].into_iter().enumerate() {
                let (w_key, w_value) = strategy.nth_item(w_dict, index).unwrap();
                assert_eq!(crate::w_str_get_value(w_key), key);
                assert_eq!(crate::w_int_get_value(w_value), value);
                assert_eq!(
                    crate::w_int_get_value(strategy.nth_value(w_dict, index).unwrap()),
                    value
                );
            }
            assert!(strategy.nth_item(w_dict, 3).is_none());
            assert!(strategy.nth_value(w_dict, 3).is_none());

            let reversed = strategy.getiterreversed(w_dict);
            let keys: Vec<_> = reversed
                .iter()
                .map(|&(key, _)| crate::w_str_get_value(key).to_owned())
                .collect();
            assert_eq!(keys, ["c", "b", "a"]);
        }
    }

    #[test]
    fn kwargs_setdefault_keeps_string_strategy_and_redispatches_other_keys() {
        unsafe {
            install_test_hash_hooks();
            let w_dict = kwargs_with(&[("a", 10)]);
            let strategy = crate::dictmultiobject::w_dict_get_strategy(w_dict);
            let w_a = crate::w_str_new("a");
            let existing = strategy.setdefault(w_dict, w_a, crate::w_int_new(99));
            assert_eq!(crate::w_int_get_value(existing), 10);

            let w_b = crate::w_str_new("b");
            let inserted = strategy.setdefault(w_dict, w_b, crate::w_int_new(20));
            assert_eq!(crate::w_int_get_value(inserted), 20);
            assert_eq!(
                crate::dictmultiobject::w_dict_get_strategy(w_dict).strategy_kind(),
                crate::dictmultiobject::StrategyKind::Kwargs
            );

            let w_int_key = crate::w_int_new(1);
            let inserted = crate::dictmultiobject::w_dict_setdefault_checked(
                w_dict,
                w_int_key,
                crate::w_int_new(30),
            )
            .unwrap();
            assert_eq!(crate::w_int_get_value(inserted), 30);
            assert_eq!(
                crate::dictmultiobject::w_dict_get_strategy(w_dict).strategy_kind(),
                crate::dictmultiobject::StrategyKind::Object
            );
        }
    }
}
