//! RPython `rpython/rtyper/controllerentry.py`.
//!
//! Controllers are an rtyper-level indirection layer: annotation sees a
//! `SomeControlledInstance`, while lowering reveals the wrapped real object
//! repr and delegates operations to controller methods. This file lands the
//! structural surface and the `ControlledInstanceRepr` forwarding behavior.
//! The full `rtypedelegate()` hop-copy rewrite remains deferred until the
//! matching `HighLevelOp.copy()`/operation reconstruction surface is ported.
#![allow(non_snake_case)]

use std::fmt;
use std::sync::Arc;

use crate::annotator::model::{KnownType, SomeValue};
use crate::flowspace::model::{ConstValue, Constant, Hlvalue};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::pairtype::ReprClassId;
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, ReprState};
use crate::translator::rtyper::rtyper::HighLevelOp;

/// RPython `controlled_instance_box(controller, obj)`.
pub fn controlled_instance_box(controller: Arc<Controller>, obj: ConstValue) -> ControlledBox {
    ControlledBox { controller, obj }
}

/// RPython `controlled_instance_unbox(controller, obj)`.
pub fn controlled_instance_unbox(
    controller: &Arc<Controller>,
    obj: &ControlledBox,
) -> Result<ConstValue, TyperError> {
    if Arc::ptr_eq(controller, &obj.controller) {
        Ok(obj.obj.clone())
    } else {
        Err(TyperError::message(
            "unbox() called with a different controller",
        ))
    }
}

/// RPython `controlled_instance_is_box(controller, obj)`.
pub fn controlled_instance_is_box(controller: &Arc<Controller>, obj: &ControlledBox) -> bool {
    Arc::ptr_eq(controller, &obj.controller)
}

#[derive(Clone)]
pub struct Controller {
    pub name: String,
    pub knowntype: KnownType,
    pub can_be_None: bool,
    convert: Arc<dyn Fn(&ConstValue) -> ConstValue + Send + Sync>,
}

impl Controller {
    pub fn new(
        name: impl Into<String>,
        knowntype: KnownType,
        can_be_None: bool,
        convert: Arc<dyn Fn(&ConstValue) -> ConstValue + Send + Sync>,
    ) -> Self {
        Controller {
            name: name.into(),
            knowntype,
            can_be_None,
            convert,
        }
    }

    pub fn identity(name: impl Into<String>, knowntype: KnownType) -> Self {
        Controller::new(name, knowntype, false, Arc::new(|value| value.clone()))
    }

    /// RPython `Controller.convert(value)` hook. The base upstream class leaves
    /// `convert` to concrete controllers; the identity constructor is the Rust
    /// equivalent for controllers that do not transform constants.
    pub fn convert(&self, value: &ConstValue) -> ConstValue {
        (self.convert)(value)
    }

    pub fn getattr(&self, _obj: &SomeValue, _attr: &str) -> Result<SomeValue, TyperError> {
        Err(TyperError::missing_rtype_operation(
            "controllerentry.Controller.getattr delegate deferred",
        ))
    }

    pub fn setattr(
        &self,
        _obj: &SomeValue,
        _attr: &str,
        _value: &SomeValue,
    ) -> Result<(), TyperError> {
        Err(TyperError::missing_rtype_operation(
            "controllerentry.Controller.setattr delegate deferred",
        ))
    }
}

impl fmt::Debug for Controller {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Controller")
            .field("name", &self.name)
            .field("knowntype", &self.knowntype)
            .field("can_be_None", &self.can_be_None)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug)]
pub struct ControlledBox {
    pub controller: Arc<Controller>,
    pub obj: ConstValue,
}

/// RPython `class ControllerEntry(ExtRegistryEntry)`.
#[derive(Clone, Debug, Default)]
pub struct ControllerEntry;

/// RPython `class ControllerEntryForPrebuilt(ExtRegistryEntry)`.
#[derive(Clone, Debug, Default)]
pub struct ControllerEntryForPrebuilt;

/// RPython `class BoxEntry(ExtRegistryEntry)`.
#[derive(Clone, Debug, Default)]
pub struct BoxEntry;

/// RPython `class UnboxEntry(ExtRegistryEntry)`.
#[derive(Clone, Debug, Default)]
pub struct UnboxEntry;

/// RPython `class IsBoxEntry(ExtRegistryEntry)`.
#[derive(Clone, Debug, Default)]
pub struct IsBoxEntry;

/// RPython `class SomeControlledInstance(annmodel.SomeObject)`.
#[derive(Clone, Debug)]
pub struct SomeControlledInstance {
    pub s_real_obj: SomeValue,
    pub controller: Arc<Controller>,
    pub knowntype: KnownType,
}

impl SomeControlledInstance {
    pub fn new(s_real_obj: SomeValue, controller: Arc<Controller>) -> Self {
        SomeControlledInstance {
            knowntype: controller.knowntype,
            s_real_obj,
            controller,
        }
    }

    pub fn can_be_none(&self) -> bool {
        self.controller.can_be_None
    }

    pub fn noneify(&self) -> Self {
        SomeControlledInstance::new(self.s_real_obj.clone(), self.controller.clone())
    }

    pub fn rtyper_makekey(&self) -> (ReprClassId, Arc<Controller>) {
        (ReprClassId::ControlledInstanceRepr, self.controller.clone())
    }
}

/// RPython `class ControlledInstanceRepr(Repr)`.
#[derive(Debug)]
pub struct ControlledInstanceRepr {
    pub s_real_obj: SomeValue,
    pub r_real_obj: Arc<dyn Repr>,
    pub controller: Arc<Controller>,
    lowleveltype: crate::translator::rtyper::lltypesystem::lltype::LowLevelType,
    state: ReprState,
}

impl ControlledInstanceRepr {
    pub fn new(
        s_real_obj: SomeValue,
        r_real_obj: Arc<dyn Repr>,
        controller: Arc<Controller>,
    ) -> Self {
        ControlledInstanceRepr {
            s_real_obj,
            lowleveltype: r_real_obj.lowleveltype().clone(),
            r_real_obj,
            controller,
            state: ReprState::new(),
        }
    }

    /// RPython `ControlledInstanceRepr.reveal(self, r)`.
    pub fn reveal<'a>(
        &'a self,
        r: &'a dyn Repr,
    ) -> Result<(&'a SomeValue, &'a Arc<dyn Repr>), TyperError> {
        let Some(other) = (r as &dyn std::any::Any).downcast_ref::<ControlledInstanceRepr>() else {
            return Err(TyperError::message(format!(
                "expected {}, got {}",
                self.repr_string(),
                r.repr_string()
            )));
        };
        if !std::ptr::eq(self, other) {
            return Err(TyperError::message(format!(
                "expected {}, got {}",
                self.repr_string(),
                r.repr_string()
            )));
        }
        Ok((&self.s_real_obj, &self.r_real_obj))
    }

    pub fn rtype_getattr(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(TyperError::missing_rtype_operation(
            "controllerentry.ControlledInstanceRepr.rtype_getattr delegate deferred",
        ))
    }

    pub fn rtype_setattr(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(TyperError::missing_rtype_operation(
            "controllerentry.ControlledInstanceRepr.rtype_setattr delegate deferred",
        ))
    }

    pub fn rtype_bool(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(TyperError::missing_rtype_operation(
            "controllerentry.ControlledInstanceRepr.rtype_bool delegate deferred",
        ))
    }

    pub fn rtype_simple_call(&self, _hop: &HighLevelOp) -> RTypeResult {
        Err(TyperError::missing_rtype_operation(
            "controllerentry.ControlledInstanceRepr.rtype_simple_call delegate deferred",
        ))
    }
}

impl Repr for ControlledInstanceRepr {
    fn lowleveltype(&self) -> &crate::translator::rtyper::lltypesystem::lltype::LowLevelType {
        &self.lowleveltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "ControlledInstanceRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::ControlledInstanceRepr
    }

    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        let real_value = self.controller.convert(value);
        self.r_real_obj.convert_const(&real_value)
    }
}

/// RPython `delegate(boundmethod, *args_s)`.
pub fn delegate(_boundmethod: &str, _args_s: &[SomeValue]) -> Result<SomeValue, TyperError> {
    Err(TyperError::missing_rtype_operation(
        "controllerentry.delegate bookkeeper emulation deferred",
    ))
}

/// RPython `rtypedelegate(callable, hop, revealargs=[0], revealresult=False)`.
pub fn rtypedelegate(
    _callable: &str,
    _hop: &HighLevelOp,
    _revealargs: &[usize],
    _revealresult: bool,
) -> Result<Option<Hlvalue>, TyperError> {
    Err(TyperError::missing_rtype_operation(
        "controllerentry.rtypedelegate HighLevelOp rewrite deferred",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::model::{SomeInteger, SomeValue};
    use crate::flowspace::model::ConstValue;
    use crate::translator::rtyper::rint::signed_repr;
    use crate::translator::rtyper::rmodel::Repr;

    #[test]
    fn controlled_repr_convert_const_delegates_after_controller_conversion() {
        let controller = Arc::new(Controller::new(
            "plus-one",
            KnownType::Int,
            false,
            Arc::new(|value| match value {
                ConstValue::Int(n) => ConstValue::Int(n + 1),
                other => other.clone(),
            }),
        ));
        let repr = ControlledInstanceRepr::new(
            SomeValue::Integer(SomeInteger::default()),
            signed_repr() as Arc<dyn Repr>,
            controller,
        );

        let converted = repr.convert_const(&ConstValue::Int(41)).unwrap();
        assert_eq!(converted.value, ConstValue::Int(42));
        assert_eq!(converted.concretetype, Some(repr.lowleveltype().clone()));
        assert_eq!(repr.repr_class_id(), ReprClassId::ControlledInstanceRepr);
        assert_eq!(
            ReprClassId::ControlledInstanceRepr.mro(),
            &[ReprClassId::ControlledInstanceRepr, ReprClassId::Repr]
        );
    }

    #[test]
    fn controlled_box_unbox_checks_controller_identity() {
        let controller = Arc::new(Controller::identity("identity", KnownType::Int));
        let other = Arc::new(Controller::identity("other", KnownType::Int));
        let boxed = controlled_instance_box(controller.clone(), ConstValue::Int(7));

        assert!(controlled_instance_is_box(&controller, &boxed));
        assert!(!controlled_instance_is_box(&other, &boxed));
        assert_eq!(
            controlled_instance_unbox(&controller, &boxed).unwrap(),
            ConstValue::Int(7)
        );
        assert!(controlled_instance_unbox(&other, &boxed).is_err());
    }

    #[test]
    fn reveal_rejects_different_controlled_repr() {
        let controller = Arc::new(Controller::identity("identity", KnownType::Int));
        let left = ControlledInstanceRepr::new(
            SomeValue::Integer(SomeInteger::default()),
            signed_repr() as Arc<dyn Repr>,
            controller.clone(),
        );
        let right = ControlledInstanceRepr::new(
            SomeValue::Integer(SomeInteger::default()),
            signed_repr() as Arc<dyn Repr>,
            controller,
        );

        assert!(left.reveal(&left).is_ok());
        let err = left.reveal(&right).unwrap_err();
        assert!(format!("{err}").contains("expected"));
    }
}
