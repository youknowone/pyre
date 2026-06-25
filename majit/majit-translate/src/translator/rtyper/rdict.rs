//! RPython `rpython/rtyper/rdict.py` — abstract dict repr surface.
//!
//! The upstream file contributes the `SomeDict.rtyper_makerepr`
//! extension, `AbstractDictRepr`, `rtype_newdict`, and
//! `AbstractDictIteratorRepr`. The concrete low-level data layout lives
//! in `lltypesystem/rdict.py`; pyre mirrors that split by keeping the
//! common repr state here and constructing the concrete
//! `lltypesystem` dict repr from the rtyper dispatch.

use std::rc::Rc;
use std::sync::Arc;

use crate::annotator::dictdef::DictDef;
use crate::annotator::model::SomeDict;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, ReprState};
use crate::translator::rtyper::rtyper::{HighLevelOp, RPythonTyper};

/// RPython `class AbstractDictRepr(rmodel.Repr)` (`rdict.py:35-63`).
///
/// Rust has no inheritance, so the lltypesystem [`DictRepr`] embeds this
/// state directly and forwards its `compact_repr` behavior.
pub struct AbstractDictRepr {
    pub state: ReprState,
    pub rtyper: Rc<RPythonTyper>,
    pub external_key_repr: Arc<dyn Repr>,
    pub key_repr: Arc<dyn Repr>,
    pub external_value_repr: Arc<dyn Repr>,
    pub value_repr: Arc<dyn Repr>,
    pub dictdef: DictDef,
    pub custom_eq_hash: bool,
    pub force_non_null: bool,
    pub simple_hash_eq: bool,
    pub custom_eq_hash_repr: Option<(Arc<dyn Repr>, Arc<dyn Repr>)>,
}

impl std::fmt::Debug for AbstractDictRepr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AbstractDictRepr")
            .field("key_repr", &self.key_repr.class_name())
            .field("value_repr", &self.value_repr.class_name())
            .field("custom_eq_hash", &self.custom_eq_hash)
            .field("force_non_null", &self.force_non_null)
            .field("simple_hash_eq", &self.simple_hash_eq)
            .finish()
    }
}

impl AbstractDictRepr {
    /// RPython `AbstractDictRepr.pickrepr` / `pickkeyrepr`
    /// (`rdict.py:37-44`).
    pub fn pickrepr(
        rtyper: &Rc<RPythonTyper>,
        item_repr: Arc<dyn Repr>,
        custom_eq_hash: bool,
    ) -> Result<(Arc<dyn Repr>, Arc<dyn Repr>), TyperError> {
        if custom_eq_hash {
            Ok((item_repr.clone(), item_repr))
        } else {
            crate::translator::rtyper::rclass::externalvsinternal(rtyper, item_repr, false)
        }
    }

    pub fn compact_repr(&self) -> String {
        format!(
            "DictR {} {}",
            self.key_repr.compact_repr(),
            self.value_repr.compact_repr()
        )
    }

    pub fn recast_value(&self) -> (&Arc<dyn Repr>, &Arc<dyn Repr>) {
        (&self.value_repr, &self.external_value_repr)
    }

    pub fn recast_key(&self) -> (&Arc<dyn Repr>, &Arc<dyn Repr>) {
        (&self.key_repr, &self.external_key_repr)
    }
}

/// RPython `def rtype_newdict(hop)` (`rdict.py:57-62`).
pub fn rtype_newdict(hop: &HighLevelOp) -> RTypeResult {
    let _ = hop;
    Err(TyperError::missing_rtype_operation(
        "rdict.rtype_newdict — ll_newdict lowering deferred to lltypesystem/rdict.py port",
    ))
}

/// RPython `class AbstractDictIteratorRepr(rmodel.IteratorRepr)`
/// (`rdict.py:66-148`).
#[derive(Debug)]
pub struct AbstractDictIteratorRepr {
    state: ReprState,
    lowleveltype: LowLevelType,
    pub variant: Vec<String>,
}

impl AbstractDictIteratorRepr {
    pub fn new(lowleveltype: LowLevelType, variant: Vec<String>) -> Self {
        AbstractDictIteratorRepr {
            state: ReprState::new(),
            lowleveltype,
            variant,
        }
    }
}

impl Repr for AbstractDictIteratorRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lowleveltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "AbstractDictIteratorRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::AbstractDictIteratorRepr
    }
}

/// RPython `SomeDict.rtyper_makerepr` (`rdict.py:12-25`).
///
/// Upstream `annotator/model.py:416` aliases `SomeDict =
/// SomeOrderedDict`, so the `SomeOrderedDict.get_dict_repr` override
/// (`rdict.py:32-34`) wins and the concrete repr is
/// `lltypesystem.rordereddict.OrderedDictRepr`.
pub(crate) fn somedict_rtyper_makerepr(
    s_dict: &SomeDict,
    rtyper: &RPythonTyper,
) -> Result<Arc<dyn Repr>, TyperError> {
    let dictkey = s_dict.dictdef.dictkey_rc();
    let dictvalue = s_dict.dictdef.dictvalue_rc();
    let s_key = dictkey.borrow().s_value.clone();
    let s_value = dictvalue.borrow().s_value.clone();
    let custom_eq_hash = dictkey.borrow().custom_eq_hash;
    let custom_eq_hash_repr = if custom_eq_hash {
        Some((
            rtyper.getrepr(&dictkey.borrow().s_rdict_eqfn)?,
            rtyper.getrepr(&dictkey.borrow().s_rdict_hashfn)?,
        ))
    } else {
        None
    };
    Ok(Arc::new(
        crate::translator::rtyper::lltypesystem::rordereddict::OrderedDictRepr::new(
            rtyper.self_rc()?,
            rtyper.getrepr(&s_key)?,
            rtyper.getrepr(&s_value)?,
            s_dict.dictdef.clone(),
            custom_eq_hash_repr,
            s_dict.dictdef.inner.force_non_null,
            s_dict.dictdef.inner.simple_hash_eq,
        )?,
    ) as Arc<dyn Repr>)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::annotator::dictdef::DictDef;
    use crate::annotator::model::{SomeDict, SomeInteger, SomeString, SomeValue};
    use crate::translator::rtyper::pairtype::ReprClassId;

    #[test]
    fn somedict_makerepr_returns_lltypesystem_dictrepr() {
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
        let s_dict = SomeDict::new(dictdef);

        let repr = somedict_rtyper_makerepr(&s_dict, &rtyper).expect("dict repr");

        assert_eq!(repr.class_name(), "OrderedDictRepr");
        assert_eq!(repr.repr_class_id(), ReprClassId::OrderedDictRepr);
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
