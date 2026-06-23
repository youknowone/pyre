//! RPython `rpython/rtyper/lltypesystem/rordereddict.py`.
//!
//! PyPy aliases all dict annotations to `SomeOrderedDict`, so this is
//! the default concrete repr selected by `SomeDict.rtyper_makerepr`.
//! This slice lands `OrderedDictRepr`'s low-level table shape and the public
//! constant/helper names around the low-level method section. The dense
//! lookup/resizing/helper family remains a follow-up line-by-line port.

use std::rc::Rc;
use std::sync::{Arc, LazyLock};

use crate::annotator::dictdef::DictDef;
use crate::flowspace::model::ConstValue;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    ArrayType, GCREF, LowLevelType, Ptr, PtrTarget, StructType,
};
use crate::translator::rtyper::rdict::AbstractDictRepr;
use crate::translator::rtyper::rmodel::{Repr, ReprState};
use crate::translator::rtyper::rtyper::RPythonTyper;

fn ptr_to_gc_array(of: LowLevelType) -> LowLevelType {
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array(ArrayType::gc(of)),
    }))
}

/// RPython `DICTINDEX_*` pointer aliases (`rordereddict.py:489-492`).
///
/// PyPy uses width-specific `rffi.UCHAR` / `USHORT` / `UINT` element types.
/// Pyre's current `lltype`/`rffi` surface does not yet distinguish those C
/// integer widths, so all four aliases carry the available unsigned primitive
/// while preserving the exact public names and `Ptr(GcArray(...))` shape.
pub static DICTINDEX_LONG: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_gc_array(LowLevelType::Unsigned));
pub static DICTINDEX_INT: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_gc_array(LowLevelType::Unsigned));
pub static DICTINDEX_SHORT: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_gc_array(LowLevelType::Unsigned));
pub static DICTINDEX_BYTE: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_gc_array(LowLevelType::Unsigned));

/// RPython `IS_64BIT = sys.maxint != 2 ** 31 - 1` (`rordereddict.py:494`).
pub const IS_64BIT: bool = usize::BITS == 64;

/// RPython lookup table selector constants (`rordereddict.py:496-503`).
pub const FUNC_SHIFT: i64 = if IS_64BIT { 3 } else { 2 };
pub const FUNC_MASK: i64 = if IS_64BIT { 0x07 } else { 0x03 };
pub const FUNC_BYTE: i64 = 0;
pub const FUNC_SHORT: i64 = 1;
pub const FUNC_INT: i64 = 2;
pub const FUNC_LONG: i64 = if IS_64BIT { 3 } else { 2 };
pub const FUNC_MUST_REINDEX: i64 = if IS_64BIT { 4 } else { 3 };

/// RPython `TYPE_*` aliases (`rordereddict.py:504-507`). See
/// [`DICTINDEX_BYTE`] for the current width-collapsing note.
pub const TYPE_BYTE: LowLevelType = LowLevelType::Unsigned;
pub const TYPE_SHORT: LowLevelType = LowLevelType::Unsigned;
pub const TYPE_INT: LowLevelType = LowLevelType::Unsigned;
pub const TYPE_LONG: LowLevelType = LowLevelType::Unsigned;

pub const PERTURB_SHIFT: i64 = 5;
pub const FREE: i64 = 0;
pub const DELETED: i64 = 1;
pub const VALID_OFFSET: i64 = 2;
pub const MIN_INDEXES_MINUS_ENTRIES: i64 = VALID_OFFSET + 1;

pub const FLAG_LOOKUP: i64 = 0;
pub const FLAG_STORE: i64 = 1;

/// RPython `DICT_INITSIZE = 16` (`rordereddict.py:1156`).
pub const DICT_INITSIZE: i64 = 16;

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

fn ordered_dict_runtime_deferred(name: &str) -> TyperError {
    TyperError::missing_rtype_operation(format!(
        "lltypesystem.rordereddict.{name} — ordered hash table runtime deferred"
    ))
}

pub fn ll_call_lookup_function() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_call_lookup_function"))
}

pub fn get_ll_dict() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("get_ll_dict"))
}

pub fn ll_no_initial_index() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_no_initial_index"))
}

pub fn ll_malloc_indexes_and_choose_lookup() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred(
        "ll_malloc_indexes_and_choose_lookup",
    ))
}

pub fn ll_clear_indexes() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_clear_indexes"))
}

pub fn _ll_write_indexes() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("_ll_write_indexes"))
}

pub fn ll_call_insert_clean_function() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred(
        "ll_call_insert_clean_function",
    ))
}

pub fn ll_call_delete_by_entry_index() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred(
        "ll_call_delete_by_entry_index",
    ))
}

pub fn ll_valid_from_flag() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_valid_from_flag"))
}

pub fn ll_valid_from_key() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_valid_from_key"))
}

pub fn ll_valid_from_value() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_valid_from_value"))
}

pub fn ll_mark_deleted_in_flag() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_mark_deleted_in_flag"))
}

pub fn ll_mark_deleted_in_key() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_mark_deleted_in_key"))
}

pub fn ll_mark_deleted_in_value() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_mark_deleted_in_value"))
}

pub fn ll_hash_from_cache() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_hash_from_cache"))
}

pub fn ll_hash_recomputed() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_hash_recomputed"))
}

pub fn ll_hash_custom_fast() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_hash_custom_fast"))
}

pub fn ll_keyhash_custom() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_keyhash_custom"))
}

pub fn ll_keyeq_custom() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_keyeq_custom"))
}

pub fn ll_dict_len() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_len"))
}

pub fn ll_dict_bool() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_bool"))
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

    fn ptr_gcarray_of(value: &LowLevelType) -> &ArrayType {
        let LowLevelType::Ptr(ptr) = value else {
            panic!("expected Ptr(GcArray), got {value:?}");
        };
        let PtrTarget::Array(array) = &ptr.TO else {
            panic!("expected Ptr(GcArray), got {value:?}");
        };
        array
    }

    #[test]
    fn lookup_constants_match_64bit_upstream_layout() {
        assert!(IS_64BIT, "test host is expected to match PyPy's 64-bit arm");
        assert_eq!(FUNC_SHIFT, 3);
        assert_eq!(FUNC_MASK, 0x07);
        assert_eq!(FUNC_BYTE, 0);
        assert_eq!(FUNC_SHORT, 1);
        assert_eq!(FUNC_INT, 2);
        assert_eq!(FUNC_LONG, 3);
        assert_eq!(FUNC_MUST_REINDEX, 4);
        assert_eq!(PERTURB_SHIFT, 5);
        assert_eq!(FREE, 0);
        assert_eq!(DELETED, 1);
        assert_eq!(VALID_OFFSET, 2);
        assert_eq!(MIN_INDEXES_MINUS_ENTRIES, 3);
        assert_eq!(FLAG_LOOKUP, 0);
        assert_eq!(FLAG_STORE, 1);
        assert_eq!(DICT_INITSIZE, 16);
    }

    #[test]
    fn dictindex_aliases_keep_ptr_gcarray_shape() {
        for alias in [
            &*DICTINDEX_BYTE,
            &*DICTINDEX_SHORT,
            &*DICTINDEX_INT,
            &*DICTINDEX_LONG,
        ] {
            assert_eq!(ptr_gcarray_of(alias).OF, LowLevelType::Unsigned);
        }
        assert_eq!(TYPE_BYTE, LowLevelType::Unsigned);
        assert_eq!(TYPE_SHORT, LowLevelType::Unsigned);
        assert_eq!(TYPE_INT, LowLevelType::Unsigned);
        assert_eq!(TYPE_LONG, LowLevelType::Unsigned);
    }

    #[test]
    fn runtime_helper_surface_is_explicitly_deferred() {
        let err = ll_call_lookup_function().expect_err("runtime helper deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_call_lookup_function"));

        let err = ll_dict_len().expect_err("runtime helper deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_dict_len"));
    }
}
