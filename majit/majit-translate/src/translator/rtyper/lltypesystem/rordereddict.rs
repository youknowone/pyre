//! RPython `rpython/rtyper/lltypesystem/rordereddict.py`.
//!
//! PyPy aliases all dict annotations to `SomeOrderedDict`, so this is
//! the default concrete repr selected by `SomeDict.rtyper_makerepr`.
//! This slice lands `OrderedDictRepr`'s low-level table shape. The dense
//! lookup/resizing/helper family remains a follow-up line-by-line port.

use std::rc::Rc;
use std::sync::Arc;

use crate::annotator::dictdef::DictDef;
use crate::flowspace::model::ConstValue;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    ArrayType, GCREF, LowLevelType, Ptr, PtrTarget, StructType,
};
use crate::translator::rtyper::rdict::AbstractDictRepr;
use crate::translator::rtyper::rmodel::{Repr, ReprState};
use crate::translator::rtyper::rtyper::RPythonTyper;

/// RPython lookup table selector constants (`rordereddict.py` around the
/// lookup helper family). Values are not semantically important until the
/// lookup helpers land, but their names are part of the port surface.
pub const FUNC_BYTE: i64 = 0;
pub const FUNC_SHORT: i64 = 1;
pub const FUNC_INT: i64 = 2;
pub const FUNC_LONG: i64 = 3;
pub const FUNC_MASK: i64 = 3;

/// RPython `class OrderedDictRepr(AbstractDictRepr)`
/// (`lltypesystem/rordereddict.py:173`).
#[derive(Debug)]
#[allow(non_snake_case)]
pub struct OrderedDictRepr {
    pub base: AbstractDictRepr,
    pub DICTKEY: LowLevelType,
    pub DICTVALUE: LowLevelType,
    /// RPython `Struct("odictentry", ...)`.
    pub DICTENTRY: StructType,
    pub DICTENTRYARRAY: ArrayType,
    /// RPython `GcStruct("dicttable", ...)`.
    pub DICT: StructType,
    lowleveltype: LowLevelType,
}

impl OrderedDictRepr {
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
        let (external_value_repr, value_repr) = if custom_eq_hash {
            (value_repr.clone(), value_repr)
        } else {
            crate::translator::rtyper::rclass::externalvsinternal(&rtyper, value_repr, true)?
        };
        let dictkey_lltype = key_repr.lowleveltype().clone();
        let dictvalue_lltype = value_repr.lowleveltype().clone();

        let mut entryfields = vec![
            ("key".into(), dictkey_lltype.clone()),
            ("f_valid".into(), LowLevelType::Bool),
            ("value".into(), dictvalue_lltype.clone()),
        ];
        if !simple_hash_eq {
            entryfields.push(("f_hash".into(), LowLevelType::Signed));
        }
        let dictentry = StructType::new("odictentry", entryfields);
        let dictentryarray = ArrayType::gc(LowLevelType::Struct(Box::new(dictentry.clone())));
        let entries_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(dictentryarray.clone()),
        }));
        let mut fields = vec![
            ("num_live_items".into(), LowLevelType::Signed),
            ("num_ever_used_items".into(), LowLevelType::Signed),
            ("resize_counter".into(), LowLevelType::Signed),
            ("indexes".into(), GCREF.clone()),
            ("lookup_function_no".into(), LowLevelType::Signed),
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

        Ok(OrderedDictRepr {
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

impl Repr for OrderedDictRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lowleveltype
    }

    fn state(&self) -> &ReprState {
        &self.base.state
    }

    fn class_name(&self) -> &'static str {
        "OrderedDictRepr"
    }

    fn repr_class_id(&self) -> crate::translator::rtyper::pairtype::ReprClassId {
        crate::translator::rtyper::pairtype::ReprClassId::OrderedDictRepr
    }

    fn compact_repr(&self) -> String {
        self.base.compact_repr()
    }
}

pub fn ll_newdict_size(_dict: &StructType, _length_estimate: usize) -> Result<(), TyperError> {
    Err(TyperError::missing_rtype_operation(
        "lltypesystem.rordereddict.ll_newdict_size — ordered hash table allocation deferred",
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
    fn ordereddictrepr_builds_sparse_index_dicttable_shape() {
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

        let repr = OrderedDictRepr::new(
            rtyper,
            signed_repr() as Arc<dyn Repr>,
            string_repr() as Arc<dyn Repr>,
            dictdef,
            None,
            false,
            false,
        )
        .expect("ordered dict repr");

        assert_eq!(repr.repr_class_id(), ReprClassId::OrderedDictRepr);
        assert_eq!(repr.DICTENTRY._name, "odictentry");
        assert_eq!(
            repr.DICTENTRY._names,
            vec!["key", "f_valid", "value", "f_hash"]
        );
        assert_eq!(
            repr.DICT._names,
            vec![
                "num_live_items",
                "num_ever_used_items",
                "resize_counter",
                "indexes",
                "lookup_function_no",
                "entries"
            ]
        );
        assert_eq!(
            ReprClassId::OrderedDictRepr.mro(),
            &[
                ReprClassId::OrderedDictRepr,
                ReprClassId::AbstractDictRepr,
                ReprClassId::Repr
            ]
        );
    }
}
