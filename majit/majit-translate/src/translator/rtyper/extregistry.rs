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

use super::error::TyperError;
use super::lltypesystem::lltype;
use super::rbuiltin::BuiltinTyperFn;

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

    /// rbuiltin.py:81 `entry.specialize_call` attribute lookup.
    ///
    /// Upstream `ExtRegistryEntry` (extregistry.py:33-72) does not
    /// define a base `specialize_call` method; subclasses that override
    /// it return a typer callable that `findbltintyper` returns
    /// directly, while subclasses that do not override it raise
    /// `AttributeError` at the attribute lookup site. The Rust enum
    /// mirrors per-variant: each arm whose upstream subclass overrides
    /// `specialize_call` returns `Ok(typer_fn)`; arms whose upstream
    /// subclass does not override it surface the AttributeError as a
    /// `TyperError` so the rtyper fails closed at the same point.
    pub fn specialize_call(&self) -> Result<BuiltinTyperFn, TyperError> {
        match self {
            // lltype.py:1513-1518 `_ptrEntry(ExtRegistryEntry)` defines
            // only `_type_` and `compute_annotation` — no
            // `specialize_call`. Upstream raises AttributeError at
            // `entry.specialize_call`.
            ExtRegistryEntry::Ptr(_) => Err(TyperError::message(
                "'ExtRegistryEntry' object has no attribute 'specialize_call'",
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
