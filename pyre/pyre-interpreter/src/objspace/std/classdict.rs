//! `pypy/objspace/std/classdict.py` — `ClassDictStrategy`.
//!
//! Exposes a `W_TypeObject` namespace as a dict. `dstorage` erases the type
//! (`strategy.erase(self)`); get/set/del go through `getdictvalue` /
//! `setdictvalue` / `deldictvalue`.

use crate::PyError;
use crate::baseobjspace::{SpaceCacheClass, SpaceCacheInstance};
use pyre_object::dictmultiobject::{DictStrategy, DictStrategyRef, StrategyKind};
use pyre_object::*;
use rustpython_wtf8::Wtf8;

/// `classdict.py ClassDictStrategy`.
pub struct ClassDictStrategy {
    #[allow(dead_code)]
    space: crate::baseobjspace::SpaceHandle,
    /// The holder `W_DictObject.dstrategy` points at. A field of the
    /// fromcache / prebuilt-space instance, not a naked module-static read.
    pub strategy_ref: &'static DictStrategyRef,
}

impl ClassDictStrategy {
    pub const fn new(space: crate::baseobjspace::SpaceHandle) -> Self {
        Self {
            space,
            strategy_ref: &CLASS_DICT_STRATEGY_REF,
        }
    }

    pub fn walk_roots(&self, _forward: &mut dyn FnMut(&mut PyObjectRef)) {}
}

/// Trait-object slot used as a dict's `dstrategy`. `fromcache` still owns a
/// per-space [`ClassDictStrategy`] so constructor identity matches upstream.
struct ClassDictStrategySlot;

impl DictStrategy for ClassDictStrategySlot {
    fn strategy_kind(&self) -> StrategyKind {
        StrategyKind::Class
    }

    fn get_empty_storage(&self) -> *mut u8 {
        std::ptr::null_mut()
    }

    unsafe fn getitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> Option<PyObjectRef> {
        CLASS_DICT_METHODS.getitem(w_dict, w_key)
    }

    unsafe fn getitem_str(&self, w_dict: PyObjectRef, key: &str) -> Option<PyObjectRef> {
        CLASS_DICT_METHODS.getitem_str(w_dict, key)
    }

    unsafe fn setitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef, w_value: PyObjectRef) {
        CLASS_DICT_METHODS.setitem(w_dict, w_key, w_value)
    }

    unsafe fn setitem_str(&self, w_dict: PyObjectRef, key: &str, w_value: PyObjectRef) {
        CLASS_DICT_METHODS.setitem_str(w_dict, key, w_value)
    }

    unsafe fn delitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> bool {
        CLASS_DICT_METHODS.delitem(w_dict, w_key)
    }

    unsafe fn length(&self, w_dict: PyObjectRef) -> usize {
        CLASS_DICT_METHODS.length(w_dict)
    }

    unsafe fn w_keys(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        CLASS_DICT_METHODS.w_keys(w_dict)
    }

    unsafe fn values(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        CLASS_DICT_METHODS.values(w_dict)
    }

    unsafe fn items(&self, w_dict: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
        CLASS_DICT_METHODS.items(w_dict)
    }

    unsafe fn clear(&self, w_dict: PyObjectRef) {
        CLASS_DICT_METHODS.clear(w_dict)
    }

    unsafe fn switch_to_object_strategy(&self, _w_dict: PyObjectRef) {}

    unsafe fn walk_gc_refs(&self, w_dict: PyObjectRef, visitor: &mut dyn FnMut(*mut PyObjectRef)) {
        CLASS_DICT_METHODS.walk_gc_refs(w_dict, visitor)
    }
}

static CLASS_DICT_SLOT: ClassDictStrategySlot = ClassDictStrategySlot;

/// The [`DictStrategyRef`] holder a class-dict's `dstrategy` slot points at.
pub static CLASS_DICT_STRATEGY_REF: DictStrategyRef = DictStrategyRef {
    kind: pyre_object::dictmultiobject::StrategyKind::Class,
    imp: &CLASS_DICT_SLOT,
    owner: std::ptr::null_mut(),
};

struct ClassDictMethods;

static CLASS_DICT_METHODS: ClassDictMethods = ClassDictMethods;

/// rerased unerase (`classdict.py`): `dstorage` is the type.
unsafe fn unerase(w_dict: PyObjectRef) -> PyObjectRef {
    (*(w_dict as *const pyre_object::W_DictObject)).dstorage as PyObjectRef
}

impl ClassDictMethods {
    unsafe fn getitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> Option<PyObjectRef> {
        if pyre_object::is_str(w_key) {
            return self.getitem_wtf8(w_dict, pyre_object::w_str_get_wtf8(w_key));
        }
        // [3.14-spec] type() may leave a non-string key in the type
        // namespace. ClassDictStrategy.getitem returns None for non-text
        // keys; the live dict_w still has to answer a key it already stored.
        let ns = type_namespace(unerase(w_dict));
        if ns.is_null() {
            return None;
        }
        let w_value = pyre_object::w_dict_lookup(ns, w_key)?;
        Some(pyre_object::celldict::unwrap_cell(w_value))
    }

    unsafe fn getitem_str(&self, w_dict: PyObjectRef, key: &str) -> Option<PyObjectRef> {
        self.getitem_wtf8(w_dict, Wtf8::new(key))
    }

    unsafe fn getitem_wtf8(&self, w_dict: PyObjectRef, key: &Wtf8) -> Option<PyObjectRef> {
        let w_type = unerase(w_dict);
        let w_value = crate::type_dict_lookup_wtf8(w_type, key)?;
        Some(pyre_object::celldict::unwrap_cell(w_value))
    }

    unsafe fn setitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef, w_value: PyObjectRef) {
        if pyre_object::is_exact_type(w_key, &pyre_object::STR_TYPE) {
            self.setitem_wtf8(w_dict, pyre_object::w_str_get_wtf8(w_key), w_value);
            return;
        }
        crate::call::set_call_error(PyError::type_error(
            "cannot add non-string keys to dict of a type",
        ));
    }

    unsafe fn setitem_str(&self, w_dict: PyObjectRef, key: &str, w_value: PyObjectRef) {
        self.setitem_wtf8(w_dict, Wtf8::new(key), w_value)
    }

    unsafe fn setitem_wtf8(&self, w_dict: PyObjectRef, key: &Wtf8, w_value: PyObjectRef) {
        // `setitem_str` additionally catches the TypeError and stores raw
        // into `dict_w` when `w_type.is_cpytype()`; pyre has no cpytype
        // concept, so the refusal propagates unconditionally.
        if let Err(err) = type_setdictvalue_wtf8(unerase(w_dict), key, w_value) {
            crate::call::set_call_error(err);
        }
    }

    unsafe fn delitem(&self, w_dict: PyObjectRef, w_key: PyObjectRef) -> bool {
        if !pyre_object::is_exact_type(w_key, &pyre_object::STR_TYPE) {
            return false;
        }
        match type_deldictvalue_wtf8(unerase(w_dict), pyre_object::w_str_get_wtf8(w_key)) {
            Ok(removed) => removed,
            Err(err) => {
                crate::call::set_call_error(err);
                false
            }
        }
    }

    unsafe fn length(&self, w_dict: PyObjectRef) -> usize {
        let ns = type_namespace(unerase(w_dict));
        if ns.is_null() {
            0
        } else {
            unsafe { pyre_object::dictmultiobject::w_dict_len(ns) }
        }
    }

    unsafe fn w_keys(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        let ns = type_namespace(unerase(w_dict));
        if ns.is_null() {
            Vec::new()
        } else {
            pyre_object::dictmultiobject::w_dict_get_strategy(ns).w_keys(ns)
        }
    }

    unsafe fn values(&self, w_dict: PyObjectRef) -> Vec<PyObjectRef> {
        let ns = type_namespace(unerase(w_dict));
        if ns.is_null() {
            return Vec::new();
        }
        pyre_object::dictmultiobject::w_dict_get_strategy(ns)
            .values(ns)
            .into_iter()
            .map(|value| pyre_object::celldict::unwrap_cell(value))
            .collect()
    }

    unsafe fn items(&self, w_dict: PyObjectRef) -> Vec<(PyObjectRef, PyObjectRef)> {
        let ns = type_namespace(unerase(w_dict));
        if ns.is_null() {
            return Vec::new();
        }
        pyre_object::dictmultiobject::w_dict_get_strategy(ns)
            .items(ns)
            .into_iter()
            .map(|(key, value)| (key, pyre_object::celldict::unwrap_cell(value)))
            .collect()
    }

    unsafe fn clear(&self, w_dict: PyObjectRef) {
        let w_type = unerase(w_dict);
        if !pyre_object::w_type_is_heaptype(w_type) {
            crate::call::set_call_error(PyError::type_error(format!(
                "can't clear dictionary of type '{}'",
                pyre_object::w_type_get_name(w_type),
            )));
            return;
        }
        let ns = type_namespace(w_type);
        if !ns.is_null() {
            pyre_object::dictmultiobject::w_dict_get_strategy(ns).clear(ns);
        }
        crate::baseobjspace::mutated(w_type, None);
    }

    unsafe fn walk_gc_refs(&self, w_dict: PyObjectRef, visitor: &mut dyn FnMut(*mut PyObjectRef)) {
        let dstorage_field =
            std::ptr::addr_of_mut!((*(w_dict as *mut pyre_object::W_DictObject)).dstorage)
                as *mut PyObjectRef;
        if (*dstorage_field).is_null() {
            return;
        }
        visitor(dstorage_field);
    }
}

fn type_namespace(w_type: PyObjectRef) -> PyObjectRef {
    let ptr = unsafe { pyre_object::w_type_get_dict_ptr(w_type) };
    if ptr.is_null() {
        PY_NULL
    } else {
        ptr as PyObjectRef
    }
}

/// `W_TypeObject.setdictvalue`.
///
/// When the type has a version tag, `write_cell` (`typeobject.py`) either
/// updates an existing `MutableCell` in place and returns `None`, or returns
/// the object the namespace must store (the raw value on the first write, a
/// fresh cell once the previous value cannot absorb the new one).  `None`
/// skips `mutated()`, so `_version_tag` does not move and a quasi-immutable
/// watcher on that field stays valid.  A type with no tag stores the raw
/// value and always mutates, matching the untagged arm.
///
/// `dont_look_inside`: the version read is the quasi-immutable field a
/// traced load already watches.  Tracing it again inside the store plants
/// a second watcher that this same store's `mutated()` revokes.
#[majit_macros::dont_look_inside]
pub(crate) unsafe fn type_setdictvalue_wtf8(
    w_type: PyObjectRef,
    name: &Wtf8,
    mut w_value: PyObjectRef,
) -> Result<(), PyError> {
    if !pyre_object::w_type_is_heaptype(w_type) {
        return Err(PyError::type_error(format!(
            "cannot set '{}' attribute of immutable type '{}'",
            name.as_str().unwrap_or("\\ud800"),
            pyre_object::w_type_get_name(w_type),
        )));
    }
    // `version_tag()` is `None` when the field is 0.
    let version_tag = crate::baseobjspace::w_type_version_tag(w_type);
    if version_tag != 0 {
        // `W_TypeObject.setdictvalue` reads through
        // `_pure_getdictvalue_no_unwrapping`, which does not unwrap.
        let w_name = pyre_object::unicodeobject::box_str_constant(name);
        let raw = crate::baseobjspace::_pure_getdictvalue_no_unwrapping(
            w_type, w_name, version_tag,
        );
        let w_curr = if raw.is_null() { None } else { Some(raw) };
        // `write_cell` returns `None` for three different stores: an
        // `IntMutableCell` updated in place, an `ObjectMutableCell` updated
        // in place, and the same object stored again.  Only the int cell
        // may skip `mutated()`.  Its payload is read with `getfield`, so the
        // version tag has to stay put.  The other two keep the tag moving:
        // method folds bake the unwrapped function under it, and a repeated
        // store of one object used to bump the tag on every assignment.
        let inplace_int = w_curr.is_some_and(|cell| {
            pyre_object::celldict::is_int_mutable_cell(cell)
        });
        match pyre_object::celldict::write_cell(w_curr, w_value) {
            None => {
                if !inplace_int {
                    crate::baseobjspace::mutated(w_type, name.as_str().ok());
                }
                return Ok(());
            }
            // An `ObjectMutableCell` would sit in the namespace where a later
            // reader that does not go through `unwrap_cell` (a type-parameter
            // bound, for one) would observe the cell.  Keep that value raw.
            // The int cell is the one an in-place update has to absorb.
            Some(stored) if pyre_object::celldict::is_object_mutable_cell(stored) => {}
            Some(stored) => w_value = stored,
        }
    }
    crate::baseobjspace::mutated(w_type, name.as_str().ok());
    crate::type_dict_store_wtf8(w_type, name, w_value);
    Ok(())
}

/// `W_TypeObject.deldictvalue`.
unsafe fn type_deldictvalue_wtf8(w_type: PyObjectRef, name: &Wtf8) -> Result<bool, PyError> {
    if !pyre_object::w_type_is_heaptype(w_type) {
        return Err(PyError::type_error(format!(
            "cannot delete attributes on immutable type object '{}'",
            pyre_object::w_type_get_name(w_type),
        )));
    }
    let removed = crate::type_dict_delete_wtf8(w_type, name);
    if removed {
        crate::baseobjspace::mutated(w_type, name.as_str().ok());
    }
    Ok(removed)
}

/// `W_TypeObject.getdict` — `space.fromcache(ClassDictStrategy)` then
/// `W_DictObject(space, strategy, strategy.erase(self))`.
pub fn class_dict_for_type(w_type: PyObjectRef) -> PyObjectRef {
    let space = crate::baseobjspace::object_space();
    pyre_object::w_dict_new_with(space.class_dict_strategy().strategy_ref, w_type as *mut u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baseobjspace::ObjSpace;
    use indexmap::IndexMap;
    use pyre_object::typedef::{TypeDef, TypeDefValue};

    #[test]
    fn fromcache_returns_one_class_dict_strategy_per_space() {
        crate::typedef::init_typeobjects();
        let a = ObjSpace::new();
        let b = ObjSpace::new();
        let SpaceCacheInstance::ClassDictStrategy(first) =
            a.fromcache(SpaceCacheClass::ClassDictStrategy)
        else {
            panic!("expected ClassDictStrategy");
        };
        let SpaceCacheInstance::ClassDictStrategy(again) =
            a.fromcache(SpaceCacheClass::ClassDictStrategy)
        else {
            panic!("expected ClassDictStrategy");
        };
        assert!(crate::baseobjspace::RetainedSpaceCache::ptr_eq(
            &first, &again
        ));
        let SpaceCacheInstance::ClassDictStrategy(other) =
            b.fromcache(SpaceCacheClass::ClassDictStrategy)
        else {
            panic!("expected ClassDictStrategy");
        };
        assert!(!crate::baseobjspace::RetainedSpaceCache::ptr_eq(
            &first, &other
        ));
    }

    #[test]
    fn type_getdict_uses_class_dict_strategy_and_reads_live_namespace() {
        crate::typedef::init_typeobjects();
        let list_type = crate::typedef::gettypeobject(&pyre_object::LIST_TYPE);
        let w_dict = crate::baseobjspace::getdict(list_type).unwrap();
        unsafe {
            assert!(!pyre_object::is_dict_proxy(w_dict));
            assert_eq!(
                pyre_object::w_dict_get_strategy(w_dict).strategy_kind(),
                StrategyKind::Class
            );
            assert!(CLASS_DICT_SLOT.getitem_str(w_dict, "append").is_some());
            let before = pyre_object::w_type_get_version_tag(list_type);
            CLASS_DICT_SLOT.setitem_str(w_dict, "append", pyre_object::w_int_new(1));
            assert!(crate::call::take_call_error().is_some());
            assert_eq!(pyre_object::w_type_get_version_tag(list_type), before);
        }
    }

    #[test]
    fn heaptype_class_dict_store_goes_through_setdictvalue() {
        crate::typedef::init_typeobjects();
        unsafe {
            let definition = TypeDef::from_rawdict(
                "HeapOwner",
                vec![],
                IndexMap::from([("__doc__".into(), TypeDefValue::Text("doc".into()))]),
                &INSTANCE_TYPE,
            );
            let w_type = ObjSpace::new().gettypeobject(definition).unwrap();
            pyre_object::w_type_set_heaptype(w_type, true);
            let w_dict = class_dict_for_type(w_type);
            let before = pyre_object::w_type_get_version_tag(w_type);
            CLASS_DICT_SLOT.setitem_str(w_dict, "attr", pyre_object::w_int_new(7));
            assert!(crate::call::take_call_error().is_none());
            assert_eq!(
                CLASS_DICT_SLOT.getitem_str(w_dict, "attr"),
                Some(crate::type_dict_lookup(w_type, "attr").unwrap())
            );
            assert_ne!(pyre_object::w_type_get_version_tag(w_type), before);
        }
    }

    #[test]
    fn class_dict_getitem_reads_surrogate_and_non_string_keys() {
        crate::typedef::init_typeobjects();
        unsafe {
            let definition =
                TypeDef::from_rawdict("SurrogateOwner", vec![], IndexMap::new(), &INSTANCE_TYPE);
            let w_type = ObjSpace::new().gettypeobject(definition).unwrap();
            pyre_object::w_type_set_heaptype(w_type, true);
            let w_dict = class_dict_for_type(w_type);
            let mut name = rustpython_wtf8::Wtf8Buf::new();
            name.push(rustpython_wtf8::CodePoint::from_u32(0xDCFF).unwrap());
            let w_name = pyre_object::w_str_from_wtf8(name);
            CLASS_DICT_SLOT.setitem(w_dict, w_name, pyre_object::w_int_new(1));
            assert!(crate::call::take_call_error().is_none());
            assert_eq!(
                CLASS_DICT_SLOT
                    .getitem(w_dict, w_name)
                    .map(|value| pyre_object::w_int_get_value(value)),
                Some(1)
            );
            let ns = type_namespace(w_type);
            pyre_object::w_dict_store(ns, pyre_object::w_int_new(1), pyre_object::w_int_new(2));
            assert_eq!(
                CLASS_DICT_SLOT
                    .getitem(w_dict, pyre_object::w_int_new(1))
                    .map(|value| pyre_object::w_int_get_value(value)),
                Some(2)
            );
        }
    }
}
