//! RPython `rpython/rtyper/lltypesystem/rdict.py` — generic dict shape.
//!
//! This slice lands the concrete `DictRepr` data layout and names used
//! by `SomeDict.rtyper_makerepr`. The full probing/resizing helper
//! family (`ll_dict_lookup`, `ll_dict_setitem`, iterators, custom
//! eq/hash dispatch) is intentionally deferred; those helpers are large
//! and must be ported line-by-line from this module.

use std::rc::Rc;
use std::sync::Arc;

use crate::annotator::dictdef::DictDef;
use crate::flowspace::model::ConstValue;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    ArrayType, LowLevelType, Ptr, PtrTarget, StructType,
};
use crate::translator::rtyper::rdict::AbstractDictRepr;
use crate::translator::rtyper::rmodel::{Repr, ReprState};
use crate::translator::rtyper::rtyper::RPythonTyper;

/// RPython `HIGHEST_BIT` / `MASK` (`lltypesystem/rdict.py:13-14`).
pub const HIGHEST_BIT: u64 = 1_u64 << (usize::BITS - 1);
pub const MASK: u64 = HIGHEST_BIT - 1;

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
    }
}
