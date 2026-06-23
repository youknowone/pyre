//! RPython `rpython/rtyper/lltypesystem/rdict.py` — generic dict shape.
//!
//! This slice lands the concrete `DictRepr` data layout and names used
//! by `SomeDict.rtyper_makerepr`. The full probing/resizing helper
//! family (`ll_dict_lookup`, `ll_dict_setitem`, iterators, custom
//! eq/hash dispatch) is intentionally deferred; those helpers are large
//! and must be ported line-by-line from this module.

use std::rc::Rc;
use std::sync::{Arc, LazyLock};

use crate::annotator::dictdef::DictDef;
use crate::flowspace::model::ConstValue;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    ArrayType, LowLevelType, Ptr, PtrTarget, StructType,
};
use crate::translator::rtyper::rdict::{AbstractDictIteratorRepr, AbstractDictRepr};
use crate::translator::rtyper::rmodel::{Repr, ReprState};
use crate::translator::rtyper::rtyper::RPythonTyper;

/// RPython `HIGHEST_BIT` / `MASK` (`lltypesystem/rdict.py:13-14`).
pub const HIGHEST_BIT: u64 = 1_u64 << (usize::BITS - 1);
pub const MASK: u64 = HIGHEST_BIT - 1;
pub const PERTURB_SHIFT: i64 = 5;
pub const DICT_INITSIZE: i64 = 8;
pub static POPITEMINDEX: LazyLock<LowLevelType> = LazyLock::new(|| {
    LowLevelType::Struct(Box::new(StructType::new(
        "PopItemIndex",
        vec![("nextindex".into(), LowLevelType::Signed)],
    )))
});
#[allow(non_upper_case_globals)]
pub static global_popitem_index: LazyLock<LowLevelType> = LazyLock::new(|| POPITEMINDEX.clone());

/// RPython `class DictRepr(AbstractDictRepr)` (`lltypesystem/rdict.py:35`).
#[derive(Debug)]
#[allow(non_snake_case)]
pub struct DictRepr {
    pub base: AbstractDictRepr,
    /// RPython `self.DICTKEY`.
    pub DICTKEY: LowLevelType,
    /// RPython `self.DICTVALUE`.
    pub DICTVALUE: LowLevelType,
    /// RPython `self.DICTENTRY = Struct("dictentry", *entryfields)`.
    pub DICTENTRY: StructType,
    /// RPython `self.DICTENTRYARRAY = GcArray(self.DICTENTRY, ...)`.
    pub DICTENTRYARRAY: ArrayType,
    /// RPython `self.DICT = GcForwardReference(); self.DICT.become(...)`.
    pub DICT: StructType,
    lowleveltype: LowLevelType,
}

impl DictRepr {
    /// RPython `DictRepr.__init__` + the data-shape part of
    /// `_setup_repr` (`lltypesystem/rdict.py:37-169`).
    pub fn new(
        rtyper: Rc<RPythonTyper>,
        key_repr: Arc<dyn Repr>,
        value_repr: Arc<dyn Repr>,
        dictdef: DictDef,
        custom_eq_hash_repr: Option<(Arc<dyn Repr>, Arc<dyn Repr>)>,
        force_non_null: bool,
        simple_hash_eq: bool,
    ) -> Result<Self, TyperError> {
        let custom_eq_hash = custom_eq_hash_repr.is_some();
        let (external_key_repr, key_repr) =
            AbstractDictRepr::pickrepr(&rtyper, key_repr, custom_eq_hash)?;
        let (external_value_repr, value_repr) =
            AbstractDictRepr::pickrepr(&rtyper, value_repr, custom_eq_hash)?;
        let dictkey_lltype = key_repr.lowleveltype().clone();
        let dictvalue_lltype = value_repr.lowleveltype().clone();

        // The full upstream entry-shape decision can elide flags by using
        // dummy key/value markers. Until those marker helpers land, keep the
        // explicit flag form from the final `else` branch:
        // key, f_everused, f_valid, value, f_hash.
        let dictentry = StructType::new(
            "dictentry",
            vec![
                ("key".into(), dictkey_lltype.clone()),
                ("f_everused".into(), LowLevelType::Bool),
                ("f_valid".into(), LowLevelType::Bool),
                ("value".into(), dictvalue_lltype.clone()),
                ("f_hash".into(), LowLevelType::Signed),
            ],
        );
        let dictentryarray = ArrayType::gc(LowLevelType::Struct(Box::new(dictentry.clone())));
        let entries_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(dictentryarray.clone()),
        }));
        let mut fields = vec![
            ("num_items".into(), LowLevelType::Signed),
            ("resize_counter".into(), LowLevelType::Signed),
            ("entries".into(), entries_ptr),
        ];
        if let Some((r_rdict_eqfn, r_rdict_hashfn)) = &custom_eq_hash_repr {
            fields.push(("fnkeyeq".into(), r_rdict_eqfn.lowleveltype().clone()));
            fields.push(("fnkeyhash".into(), r_rdict_hashfn.lowleveltype().clone()));
        }
        let dict = StructType::gc_with_hints(
            "dicttable",
            fields,
            vec![("dict".into(), ConstValue::Bool(true))],
        );
        let lowleveltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(dict.clone()),
        }));

        Ok(DictRepr {
            base: AbstractDictRepr {
                state: ReprState::new(),
                rtyper,
                external_key_repr,
                key_repr,
                external_value_repr,
                value_repr,
                dictdef,
                custom_eq_hash,
                force_non_null,
                simple_hash_eq,
                custom_eq_hash_repr,
            },
            DICTKEY: dictkey_lltype,
            DICTVALUE: dictvalue_lltype,
            DICTENTRY: dictentry,
            DICTENTRYARRAY: dictentryarray,
            DICT: dict,
            lowleveltype,
        })
    }
}

impl Repr for DictRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lowleveltype
    }

    fn state(&self) -> &ReprState {
        &self.base.state
    }

    fn class_name(&self) -> &'static str {
        "DictRepr"
    }

    fn repr_class_id(&self) -> crate::translator::rtyper::pairtype::ReprClassId {
        crate::translator::rtyper::pairtype::ReprClassId::DictRepr
    }

    fn compact_repr(&self) -> String {
        self.base.compact_repr()
    }
}

/// RPython `ll_newdict_size(DICT, length_estimate)` placeholder.
pub fn ll_newdict_size(_dict: &StructType, _length_estimate: usize) -> Result<(), TyperError> {
    Err(TyperError::missing_rtype_operation(
        "lltypesystem.rdict.ll_newdict_size — hash table allocation deferred",
    ))
}

fn rdict_runtime_deferred(name: &str) -> TyperError {
    TyperError::missing_rtype_operation(format!(
        "lltypesystem.rdict.{name} — hash table runtime deferred"
    ))
}

pub fn ll_everused_from_flag() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_everused_from_flag"))
}

pub fn ll_everused_from_key() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_everused_from_key"))
}

pub fn ll_everused_from_value() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_everused_from_value"))
}

pub fn ll_valid_from_flag() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_valid_from_flag"))
}

pub fn ll_mark_deleted_in_flag() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_mark_deleted_in_flag"))
}

pub fn ll_valid_from_key() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_valid_from_key"))
}

pub fn ll_mark_deleted_in_key() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_mark_deleted_in_key"))
}

pub fn ll_valid_from_value() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_valid_from_value"))
}

pub fn ll_mark_deleted_in_value() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_mark_deleted_in_value"))
}

pub fn ll_hash_from_cache() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_hash_from_cache"))
}

pub fn ll_hash_recomputed() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_hash_recomputed"))
}

pub fn ll_get_value() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_get_value"))
}

pub fn ll_keyhash_custom() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_keyhash_custom"))
}

pub fn ll_keyeq_custom() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_keyeq_custom"))
}

pub fn ll_dict_len() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_dict_len"))
}

pub fn ll_dict_bool() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_dict_bool"))
}

pub fn ll_dict_getitem() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_dict_getitem"))
}

pub fn ll_dict_setitem() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_dict_setitem"))
}

pub fn _ll_dict_setitem_lookup_done() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("_ll_dict_setitem_lookup_done"))
}

pub fn ll_dict_insertclean() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_dict_insertclean"))
}

pub fn ll_dict_delitem() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_dict_delitem"))
}

pub fn _ll_dict_del() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("_ll_dict_del"))
}

pub fn ll_dict_resize() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_dict_resize"))
}

pub fn _ll_dict_resize_to() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("_ll_dict_resize_to"))
}

pub fn ll_dict_lookup() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_dict_lookup"))
}

pub fn ll_dict_lookup_clean() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_dict_lookup_clean"))
}

pub fn ll_newdict() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_newdict"))
}

pub fn _ll_malloc_dict() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("_ll_malloc_dict"))
}

pub fn _ll_malloc_entries() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("_ll_malloc_entries"))
}

pub fn _ll_free_entries() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("_ll_free_entries"))
}

fn get_ll_dictiter(dictptr: LowLevelType) -> LowLevelType {
    let dictiter = StructType::gc(
        "dictiter",
        vec![
            ("dict".into(), dictptr),
            ("index".into(), LowLevelType::Signed),
        ],
    );
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Struct(dictiter),
    }))
}

/// RPython `class DictIteratorRepr(AbstractDictIteratorRepr)`
/// (`lltypesystem/rdict.py:693`).
#[derive(Debug)]
pub struct DictIteratorRepr {
    pub base: AbstractDictIteratorRepr,
    pub r_dict_lowleveltype: LowLevelType,
}

impl DictIteratorRepr {
    pub fn new(r_dict_lowleveltype: LowLevelType, variant: impl Into<String>) -> Self {
        let variant = variant.into();
        let lowleveltype = get_ll_dictiter(r_dict_lowleveltype.clone());
        DictIteratorRepr {
            base: AbstractDictIteratorRepr::new(lowleveltype, vec![variant]),
            r_dict_lowleveltype,
        }
    }
}

impl Repr for DictIteratorRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        self.base.lowleveltype()
    }

    fn state(&self) -> &ReprState {
        self.base.state()
    }

    fn class_name(&self) -> &'static str {
        "DictIteratorRepr"
    }

    fn repr_class_id(&self) -> crate::translator::rtyper::pairtype::ReprClassId {
        crate::translator::rtyper::pairtype::ReprClassId::DictIteratorRepr
    }
}

pub fn ll_dictiter() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_dictiter"))
}

pub fn _ll_dictnext() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("_ll_dictnext"))
}

pub fn ll_get() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_get"))
}

pub fn ll_setdefault() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_setdefault"))
}

pub fn ll_copy() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_copy"))
}

pub fn ll_clear() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_clear"))
}

pub fn ll_update() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_update"))
}

pub fn ll_prepare_dict_update() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_prepare_dict_update"))
}

pub fn recast() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("recast"))
}

pub fn _make_ll_keys_values_items() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("_make_ll_keys_values_items"))
}

pub fn ll_dict_keys() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_dict_keys"))
}

pub fn ll_dict_values() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_dict_values"))
}

pub fn ll_dict_items() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_dict_items"))
}

pub fn ll_contains() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_contains"))
}

pub fn _ll_getnextitem() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("_ll_getnextitem"))
}

pub fn ll_popitem() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_popitem"))
}

pub fn ll_pop() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_pop"))
}

pub fn ll_pop_default() -> Result<(), TyperError> {
    Err(rdict_runtime_deferred("ll_pop_default"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::annotator::dictdef::DictDef;
    use crate::annotator::model::{SomeInteger, SomeString, SomeValue};
    use crate::translator::rtyper::pairtype::ReprClassId;
    use crate::translator::rtyper::rint::signed_repr;
    use crate::translator::rtyper::rstr::string_repr;

    #[test]
    fn dictrepr_builds_dicttable_shape() {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let dictdef = DictDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );

        let repr = DictRepr::new(
            rtyper,
            signed_repr() as Arc<dyn Repr>,
            string_repr() as Arc<dyn Repr>,
            dictdef,
            None,
            false,
            false,
        )
        .expect("dict repr");

        assert_eq!(repr.repr_class_id(), ReprClassId::DictRepr);
        assert_eq!(repr.DICT._name, "dicttable");
        assert_eq!(
            repr.DICT._names,
            vec!["num_items", "resize_counter", "entries"]
        );
        assert_eq!(
            repr.DICTENTRY._names,
            vec!["key", "f_everused", "f_valid", "value", "f_hash"]
        );
        assert_eq!(repr.DICTKEY, LowLevelType::Signed);
        assert!(matches!(repr.lowleveltype(), LowLevelType::Ptr(_)));
        assert_eq!(PERTURB_SHIFT, 5);
        assert_eq!(DICT_INITSIZE, 8);

        let LowLevelType::Struct(popitem_index) = &*POPITEMINDEX else {
            panic!("expected raw Struct, got {:?}", &*POPITEMINDEX);
        };
        assert_eq!(popitem_index._name, "PopItemIndex");
        assert_eq!(popitem_index._names, vec!["nextindex"]);
        assert_eq!(&*global_popitem_index, &*POPITEMINDEX);
    }

    #[test]
    fn dictiteratorrepr_builds_dictiter_shape() {
        let dictptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(StructType::gc("dicttable", vec![])),
        }));
        let repr = DictIteratorRepr::new(dictptr.clone(), "keys");

        assert_eq!(repr.class_name(), "DictIteratorRepr");
        assert_eq!(repr.repr_class_id(), ReprClassId::DictIteratorRepr);
        assert_eq!(repr.base.variant, vec!["keys".to_string()]);
        assert_eq!(repr.r_dict_lowleveltype, dictptr);
        assert_eq!(
            ReprClassId::DictIteratorRepr.mro(),
            &[
                ReprClassId::DictIteratorRepr,
                ReprClassId::AbstractDictIteratorRepr,
                ReprClassId::Repr
            ]
        );

        let LowLevelType::Ptr(iter_ptr) = repr.lowleveltype() else {
            panic!("expected Ptr(GcStruct), got {:?}", repr.lowleveltype());
        };
        let PtrTarget::Struct(iter) = &iter_ptr.TO else {
            panic!("expected Ptr(GcStruct), got {iter_ptr:?}");
        };
        assert_eq!(iter._name, "dictiter");
        assert_eq!(iter._names, vec!["dict", "index"]);
    }

    #[test]
    fn runtime_helper_surface_is_explicitly_deferred() {
        let err = ll_dict_lookup().expect_err("runtime helper deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_dict_lookup"));

        let err = ll_dict_keys().expect_err("runtime helper deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_dict_keys"));
    }
}
