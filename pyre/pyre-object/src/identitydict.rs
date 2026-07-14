//! `pypy/objspace/std/identitydict.py` port — strategy for custom
//! instances which compare by identity (the default unless you
//! override `__hash__`, `__eq__` or `__cmp__`).
//!
//! Selected by `EmptyDictStrategy.switch_to_correct_strategy` per
//! `dictmultiobject.py:725-730 switch_to_identity_strategy` when the
//! key's type satisfies `W_TypeObject.compares_by_identity()`
//! (`typeobject.py:353-371`).  The distinction from
//! `ObjectDictStrategy` is in the key comparison: raw PyObjectRef
//! identity (`is`) instead of `dict_keys_equal` (`space.eq_w` +
//! `space.hash_w`).

#![allow(unsafe_op_in_unsafe_fn)]

use crate::dictmultiobject::{DictStrategy, OBJECT_DICT_STRATEGY};
use crate::pyobject::PyObjectRef;

/// `identitydict.py:12-83 IdentityDictStrategy` key type — identity
/// comparison + identity hash.  Stored in
/// `IndexMap<IdentityKey, PyObjectRef>` for O(1) lookup matching
/// PyPy's `mark_dict_non_null(d={})` (`:30-32`) — RPython resolves
/// `{}` keyed on instance identity to an order-preserving identity
/// hash table at translation time.
#[derive(Clone, Copy)]
pub struct IdentityKey(pub PyObjectRef);

impl std::hash::Hash for IdentityKey {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        crate::gc_hook::gc_identity_hash(self.0 as usize).hash(state);
    }
}

impl PartialEq for IdentityKey {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.0, other.0)
    }
}

impl Eq for IdentityKey {}

/// Typed read accessor — `dictmultiobject.py:1063 IdentityDictStrategy.unerase
/// (w_dict.dstorage)`.
///
/// # Safety
/// `obj` must point to a valid `W_DictObject` on
/// [`IDENTITY_DICT_STRATEGY`].
#[inline]
unsafe fn identity_storage<'a>(
    obj: PyObjectRef,
) -> &'a indexmap::IndexMap<IdentityKey, PyObjectRef> {
    let dict = &*(obj as *const crate::dictmultiobject::W_DictObject);
    &*(dict.dstorage as *const indexmap::IndexMap<IdentityKey, PyObjectRef>)
}

#[inline]
unsafe fn identity_storage_mut<'a>(
    obj: PyObjectRef,
) -> &'a mut indexmap::IndexMap<IdentityKey, PyObjectRef> {
    let dict = &mut *(obj as *mut crate::dictmultiobject::W_DictObject);
    &mut *(dict.dstorage as *mut indexmap::IndexMap<IdentityKey, PyObjectRef>)
}

/// `identitydict.py:12-83 IdentityDictStrategy`.
///
/// ```python
/// class IdentityDictStrategy(AbstractTypedStrategy, DictStrategy):
///     erase, unerase = rerased.new_erasing_pair("identitydict")
///
///     def wrap(self, unwrapped):
///         return unwrapped
///     def unwrap(self, wrapped):
///         return wrapped
///
///     def get_empty_storage(self):
///         d = {}
///         mark_dict_non_null(d)
///         return self.erase(d)
///
///     def is_correct_type(self, w_obj):
///         w_type = self.space.type(w_obj)
///         return w_type.compares_by_identity()
///
///     def _never_equal_to(self, w_lookup_type):
///         return False
///
///     def w_keys(self, w_dict):
///         return self.space.newlist(self.unerase(w_dict.dstorage).keys())
/// ```
pub struct IdentityDictStrategy;

/// `pypy/objspace/std/identitydict.py:12 IdentityDictStrategy`
/// singleton — matches PyPy's `space.fromcache(IdentityDictStrategy)`.
pub static IDENTITY_DICT_STRATEGY: IdentityDictStrategy = IdentityDictStrategy;

impl IdentityDictStrategy {
    /// `identitydict.py:36-37 IdentityDictStrategy.is_correct_type` —
    /// `self.space.type(w_obj).compares_by_identity()`.  Dispatch
    /// through the `dict_eq_hook::COMPARES_BY_IDENTITY_HOOK`
    /// trampoline (pyre-interpreter installs the MRO walker).
    #[inline]
    unsafe fn is_correct_type(w_key: PyObjectRef) -> bool {
        // A tagged immediate is an `int` (value-compared), never a
        // compares-by-identity key; short-circuit before the `w_class`
        // deref. Gated on `CAN_BE_TAGGED` (default false).
        if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(w_key) {
            return false;
        }
        let w_type = (*w_key).w_class as PyObjectRef;
        if w_type.is_null() {
            return false;
        }
        matches!(
            crate::dict_eq_hook::try_compares_by_identity(w_type),
            Some(true)
        )
    }
}

impl DictStrategy for IdentityDictStrategy {
    fn strategy_kind(&self) -> crate::dictmultiobject::StrategyKind {
        crate::dictmultiobject::StrategyKind::Identity
    }

    /// `dictmultiobject.py:1143-1150 AbstractTypedStrategy.switch_to_object_strategy`
    /// instantiation for IdentityDictStrategy — `wrap` is identity
    /// (`:26-27`), so the migration ports each `IdentityKey(obj)` into
    /// `ObjectKey { hash: hash_w(obj), obj }` without rewrapping keys.
    unsafe fn switch_to_object_strategy(&self, w_dict: PyObjectRef) {
        let dict = &mut *(w_dict as *mut crate::dictmultiobject::W_DictObject);
        let old = Box::from_raw(dict.dstorage as *mut indexmap::IndexMap<IdentityKey, PyObjectRef>);
        let mut new_map: indexmap::IndexMap<crate::dictmultiobject::ObjectKey, PyObjectRef> =
            indexmap::IndexMap::with_capacity(old.len());
        for (k, &v) in old.iter() {
            new_map.insert(crate::dictmultiobject::object_key_for(k.0), v);
        }
        dict.dstorage = Box::into_raw(Box::new(new_map)) as *mut u8;
        dict.dstrategy = &OBJECT_DICT_STRATEGY;
    }

    /// `identitydict.py:67-70 get_empty_storage` — erased `{}` with
    /// non-null hint.  Pyre stores
    /// `IndexMap<IdentityKey, PyObjectRef>` — identity-keyed hash
    /// bucket for O(1) lookup + insertion-order preserving iteration.
    fn get_empty_storage(&self) -> *mut u8 {
        let v: Box<indexmap::IndexMap<IdentityKey, PyObjectRef>> =
            Box::new(indexmap::IndexMap::new());
        Box::into_raw(v) as *mut u8
    }

    unsafe fn dealloc_storage(&self, w_dict: PyObjectRef) {
        let dict = &*(w_dict as *const crate::dictmultiobject::W_DictObject);
        drop(Box::from_raw(
            dict.dstorage as *mut indexmap::IndexMap<IdentityKey, PyObjectRef>,
        ));
    }

    /// `dictmultiobject.py:1095-1103 AbstractTypedStrategy.getitem` —
    /// O(1) identity-keyed lookup.
    unsafe fn getitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> Option<PyObjectRef> {
        if Self::is_correct_type(w_key) {
            return identity_storage(w_dict).get(&IdentityKey(w_key)).copied();
        }
        // `identitydict.py:40-41 _never_equal_to` → always False, so
        // mismatched keys always promote and retry.
        self.switch_to_object_strategy(w_dict);
        crate::dictmultiobject::w_dict_lookup(w_dict, w_key)
    }

    /// `dictmultiobject.py:1061-1067 setitem` — identity-keyed insert;
    /// on mismatch, promote to Object.
    unsafe fn setitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef, w_value: PyObjectRef) {
        if Self::is_correct_type(w_key) {
            let entries = identity_storage_mut(w_dict);
            entries.insert(IdentityKey(w_key), w_value);
            crate::gc_hook::try_gc_write_barrier(w_dict as *mut u8);
            return;
        }
        self.switch_to_object_strategy(w_dict);
        crate::dictmultiobject::w_dict_store(w_dict, w_key, w_value);
    }

    /// `dictmultiobject.py:1069-1071 setitem_str` — IdentityDictStrategy
    /// promotes to Object on str setitem_str (str has its own
    /// non-identity `__eq__` / `__hash__`, so OVERRIDES_EQ_CMP_OR_HASH).
    unsafe fn setitem_str(&self, w_dict: PyObjectRef, key: &str, w_value: PyObjectRef) {
        self.switch_to_object_strategy(w_dict);
        crate::dictmultiobject::w_dict_setitem_str(w_dict, key, w_value);
    }

    /// `dictmultiobject.py:1081-1087 delitem` — identity-keyed remove;
    /// on mismatch, promote to Object.
    unsafe fn delitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> bool {
        if Self::is_correct_type(w_key) {
            let entries = identity_storage_mut(w_dict);
            return entries.shift_remove(&IdentityKey(w_key)).is_some();
        }
        self.switch_to_object_strategy(w_dict);
        crate::dictmultiobject::w_dict_delitem(w_dict, w_key)
    }

    unsafe fn length(&self, w_dict: PyObjectRef) -> usize {
        identity_storage(w_dict).len()
    }

    unsafe fn w_keys(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        identity_storage(w_dict).keys().map(|k| k.0).collect()
    }

    unsafe fn values(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        identity_storage(w_dict).values().copied().collect()
    }

    unsafe fn items(&self, w_dict: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
        identity_storage(w_dict)
            .iter()
            .map(|(k, &v)| (k.0, v))
            .collect()
    }

    unsafe fn clear(&self, w_dict: PyObjectRef) {
        let entries = identity_storage_mut(w_dict);
        entries.clear();
    }

    /// `dictmultiobject.py:1152 AbstractTypedStrategy.copy` — clone
    /// the typed `IndexMap<IdentityKey, _>` backing and wrap with the
    /// same IdentityDictStrategy.
    unsafe fn copy(&self, w_dict: PyObjectRef) -> PyObjectRef {
        let storage = identity_storage(w_dict);
        let new_storage = Box::into_raw(Box::new(storage.clone()));
        crate::dictmultiobject::w_dict_new_with(&IDENTITY_DICT_STRATEGY, new_storage as *mut u8)
    }

    /// `pypy.objspace.std.identitydict.IdentityDictStrategy` stores
    /// keys as `list[W_Root]` — the GC traces them as roots and follows
    /// the pointer.  Trace `key.0` through an unsafe pointer cast so a
    /// minor collection keeps each identity key's referent alive.
    ///
    /// Hash is now stable across GC moves via `gc_identity_hash`
    /// (shadow-based, minimark.py:1900-1915), so no rehash is needed.
    /// Trace key and value pointers so the GC updates them in place.
    unsafe fn walk_gc_refs(&self, w_dict: PyObjectRef, visitor: &mut dyn FnMut(*mut PyObjectRef)) {
        let entries = identity_storage_mut(w_dict);
        for (k, v) in entries.iter_mut() {
            let key_ptr = k as *const IdentityKey as *mut IdentityKey;
            visitor(std::ptr::addr_of_mut!((*key_ptr).0));
            visitor(v as *mut PyObjectRef);
        }
    }
}
