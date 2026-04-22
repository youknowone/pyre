//! Minimal `rpython/rtyper/extregistry.py` surface used by
//! `Bookkeeper.immutablevalue(x)`.
//!
//! This module currently mirrors only the `_ptr` registration from
//! `rpython/rtyper/lltypesystem/lltype.py:_ptrEntry`. The public
//! interface stays aligned with upstream (`is_registered` / `lookup` /
//! `compute_annotation_bk`) so later entries can land without changing
//! the bookkeeper call site again.

use std::rc::Rc;

use crate::annotator::bookkeeper::Bookkeeper;
use crate::annotator::model::{AnnotatorError, SomeValue};
use crate::flowspace::model::ConstValue;

use super::lltypesystem::lltype;

#[derive(Clone, Debug)]
pub enum ExtRegistryEntry {
    Ptr(lltype::_ptr),
}

impl ExtRegistryEntry {
    pub fn compute_annotation_bk(
        &self,
        _bookkeeper: &Rc<Bookkeeper>,
    ) -> Result<SomeValue, AnnotatorError> {
        self.compute_annotation()
    }

    pub fn compute_annotation(&self) -> Result<SomeValue, AnnotatorError> {
        match self {
            // upstream lltype.py:1513-1518:
            //   class _ptrEntry(ExtRegistryEntry):
            //       _type_ = _ptr
            //
            //       def compute_annotation(self):
            //           from rpython.rtyper.llannotation import SomePtr
            //           return SomePtr(typeOf(self.instance))
            ExtRegistryEntry::Ptr(instance) => Ok(super::llannotation::lltype_to_annotation(
                lltype::typeOf(instance),
            )),
        }
    }
}

pub fn is_registered(instance: &ConstValue) -> bool {
    matches!(instance, ConstValue::LLPtr(_))
}

pub fn lookup(instance: &ConstValue) -> Option<ExtRegistryEntry> {
    match instance {
        ConstValue::LLPtr(ptr) => Some(ExtRegistryEntry::Ptr((**ptr).clone())),
        _ => None,
    }
}
