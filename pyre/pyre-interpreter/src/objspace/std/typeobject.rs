//! `pypy/objspace/std/typeobject.py`: declaration-driven builtin TypeCache.

use crate::baseobjspace::{ObjSpace, SpaceCache, SpaceCacheBuild};
use majit_rlib::cache::CacheError;
use pyre_object::typedef::{TypeDef, TypeDefValue};
use pyre_object::*;

pub struct TypeCache {
    base: SpaceCache<usize, usize, std::sync::Arc<ObjSpace>>,
}

impl TypeCache {
    pub fn new(space: std::sync::Arc<ObjSpace>) -> Self {
        Self {
            base: SpaceCache::new(space),
        }
    }

    /// # Safety
    /// The definition and its bases are immutable, live prebuilt TypeDefs.
    pub unsafe fn getorbuild(
        &self,
        definition: *const TypeDef,
    ) -> Result<PyObjectRef, CacheError<crate::PyError>> {
        self.base
            .getorbuild(definition as usize, self)
            .map(|value| value as PyObjectRef)
    }

    pub fn walk_roots(&self, forward: &mut dyn FnMut(&mut PyObjectRef)) {
        self.base.visit_values_mut(|value| {
            let mut w_value = *value as PyObjectRef;
            forward(&mut w_value);
            *value = w_value as usize;
        });
    }

    /// TypeCache.build: allocate identity, compute bases, wrap declarations,
    /// initialize, then qualify functions. Cache publication/ready is outside
    /// this method, in SpaceCache's inherited getorbuild lifecycle.
    #[majit_macros::not_rpython]
    unsafe fn build(
        &self,
        definition: *const TypeDef,
    ) -> Result<PyObjectRef, CacheError<crate::PyError>> {
        let space = &self.base.space;
        let definition = unsafe { &*definition };
        let name = definition
            .name
            .as_deref()
            .expect("layout-only TypeDef has no declarations");
        let w_type = w_type_alloc_builtin();
        let _roots = gc_roots::push_roots();
        let bases_start = gc_roots::shadow_stack_len();
        if definition.bases.is_empty() {
            // The root object's declaration still belongs to the legacy
            // bootstrap. Do not infer a declaration key from its Layout.
            let object = crate::typedef::w_object();
            assert!(
                !object.is_null(),
                "root TypeDef bootstrap must precede derived declarations"
            );
            let _ = gc_roots::pin_root(object);
        } else {
            for &base in &definition.bases {
                let w_base = unsafe { space.gettypeobject(base)? };
                let _ = gc_roots::pin_root(w_base);
            }
        }
        let bases: Vec<_> = (bases_start..gc_roots::shadow_stack_len())
            .map(gc_roots::shadow_stack_get)
            .collect();
        let bases_slot = gc_roots::shadow_stack_len();
        let _ = gc_roots::pin_root(w_tuple_new(bases));
        let ns_slot = gc_roots::shadow_stack_len();
        let _ = gc_roots::pin_root(w_dict_new());
        for (name, value) in &definition.rawdict {
            let value = match value {
                TypeDefValue::Text(text) => w_str_new(text),
                TypeDefValue::None => w_none(),
                TypeDefValue::Root(slot) => {
                    let value = unsafe { *slot.get() };
                    if unsafe { pyre_object::typedef::is_getset_property(value) } {
                        crate::typedef::copy_for_type(value, w_type)
                    } else if unsafe { pyre_object::gateway::is_interp2app(value) } {
                        crate::gateway::interp2app_spacebind(value, &space)
                    } else {
                        value
                    }
                }
            };
            unsafe {
                w_dict_setitem_str_no_proxy(gc_roots::shadow_stack_get(ns_slot), name, value)
            };
        }
        if let Some(owner) = crate::typedef::method_owner(name) {
            unsafe {
                crate::typedef::stamp_method_owners(gc_roots::shadow_stack_get(ns_slot), owner)
            };
        }
        crate::typedef::init_builtin_typeobject(
            w_type,
            name,
            gc_roots::shadow_stack_get(bases_slot),
            gc_roots::shadow_stack_get(ns_slot),
            crate::typedef::w_type(),
        );
        unsafe {
            w_type_set_hasdict(w_type, definition.hasdict);
            w_type_set_weakrefable(w_type, definition.weakrefable);
            crate::baseobjspace::compute_and_set_mro(w_type).map_err(CacheError::Build)?;
            let best = crate::call::find_best_base(gc_roots::shadow_stack_get(bases_slot))
                .map_err(CacheError::Build)?;
            let parent_layout = if best.is_null() {
                std::ptr::null()
            } else {
                w_type_get_layout_ptr(best)
            };
            let layout =
                if !parent_layout.is_null() && std::ptr::eq((*parent_layout).typedef, definition) {
                    parent_layout
                } else {
                    typeobject::leak_layout(typeobject::Layout {
                        typedef: definition,
                        nslots: 0,
                        newslotnames: vec![],
                        base_layout: parent_layout,
                        dict_data_slot: typeobject::DICT_DATA_SLOT_UNRESOLVED,
                    })
                };
            w_type_set_layout(w_type, layout);
            crate::typedef::stamp_new_descr_self(gc_roots::shadow_stack_get(ns_slot), w_type);
        }
        Ok(w_type)
    }
}

impl SpaceCacheBuild<usize, usize> for TypeCache {
    type Error = crate::PyError;

    fn build(&self, key: &usize) -> Result<usize, CacheError<Self::Error>> {
        unsafe { TypeCache::build(self, *key as *const TypeDef).map(|value| value as usize) }
    }

    fn ready(&self, value: &usize) -> Result<(), CacheError<Self::Error>> {
        unsafe { typeobject::w_type_ready(*value as PyObjectRef) };
        gc_roots::mark_prebuilt_roots_dirty();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;

    fn answer(_: &[PyObjectRef]) -> crate::PyResult {
        Ok(w_int_new(42))
    }

    #[test]
    fn declarations_keep_typedef_identity_and_bind_functions_per_space() {
        crate::typedef::init_typeobjects();
        let gateway =
            crate::gateway::interp2app(crate::gateway::builtin_code_new("original", answer));
        unsafe {
            let definition = TypeDef::from_rawdict(
                "Declared",
                vec![],
                IndexMap::from([
                    ("method".into(), TypeDefValue::root(gateway)),
                    (
                        "__doc__".into(),
                        TypeDefValue::Text("declaration doc".into()),
                    ),
                ]),
                &INSTANCE_TYPE,
            );
            let gateway = &*(gateway as *const pyre_object::gateway::interp2app);
            assert_eq!(gateway.name, "method");
            assert!(gateway._is_type_method);
            let a = ObjSpace::new();
            let b = ObjSpace::new();
            let first = a.gettypeobject(definition).unwrap();
            assert_eq!(first, a.gettypeobject(definition).unwrap());
            let other = b.gettypeobject(definition).unwrap();
            assert_ne!(first, other);
            assert_eq!((*w_type_get_layout_ptr(first)).typedef, definition);
            assert_eq!((*w_type_get_layout_ptr(other)).typedef, definition);
            let first_ns = w_type_get_dict_ptr(first) as PyObjectRef;
            let other_ns = w_type_get_dict_ptr(other) as PyObjectRef;
            assert_ne!(
                w_dict_getitem_str(first_ns, "method"),
                w_dict_getitem_str(other_ns, "method")
            );
            assert_eq!(
                w_str_get_value(w_dict_getitem_str(first_ns, "__doc__").unwrap()),
                "declaration doc"
            );
            assert!(!w_type_get_mro(first).is_null());
            let child =
                TypeDef::from_rawdict("Child", vec![definition], IndexMap::new(), &INSTANCE_TYPE);
            let child = a.gettypeobject(child).unwrap();
            assert_eq!(w_tuple_getitem(w_type_get_bases(child), 0), Some(first));
        }
    }

    #[test]
    fn production_iterator_uses_the_declared_typedef_cache_entry() {
        crate::typedef::init_typeobjects();
        let definition = crate::objspace::std::iterobject::typedef();
        unsafe {
            let cached = crate::baseobjspace::object_space()
                .gettypeobject(definition)
                .unwrap();
            let registered = crate::typedef::gettypefor(&pyre_object::iterobject::SEQ_ITER_TYPE)
                .unwrap()
                .as_ptr();
            assert_eq!(cached, registered);
            assert_eq!((*w_type_get_layout_ptr(cached)).typedef, definition);
            assert!(!(*definition).acceptable_as_base_class());
        }
    }

    #[test]
    fn getset_template_is_copied_for_each_cached_type_identity() {
        crate::typedef::init_typeobjects();
        let property = pyre_object::typedef::w_getset_property_new(
            PY_NULL,
            PY_NULL,
            PY_NULL,
            PY_NULL,
            PY_NULL,
            false,
            w_str_new("template"),
        );
        unsafe {
            let definition = TypeDef::from_rawdict(
                "PropertyOwner",
                vec![],
                IndexMap::from([("field".into(), TypeDefValue::root(property))]),
                &INSTANCE_TYPE,
            );
            let a = ObjSpace::new();
            let b = ObjSpace::new();
            for space in [&a, &b] {
                let w_type = space.gettypeobject(definition).unwrap();
                let ns = w_type_get_dict_ptr(w_type) as PyObjectRef;
                let bound = w_dict_getitem_str(ns, "field").unwrap();
                assert_ne!(bound, property);
                assert_eq!(w_getset_get_objclass(bound), w_type);
            }
            assert!(w_getset_get_objclass(property).is_null());
        }
    }
}
