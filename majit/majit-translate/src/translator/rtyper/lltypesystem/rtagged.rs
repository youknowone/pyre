//! RPython `rpython/rtyper/lltypesystem/rtagged.py` parity module.
//!
//! Tagged instance repr integration still belongs in `rclass`:
//! `getinstancerepr` currently caches `Arc<InstanceRepr>`, so replacing
//! the tagged-pointer arm with a polymorphic `TaggedInstanceRepr` needs
//! that cache shape to move first. This module lands the upstream file
//! path and the low-level tagged-pointer helpers that already have
//! concrete support in `lltype`.

use std::cell::RefCell;
use std::rc::{Rc, Weak};

use crate::annotator::classdesc::ClassDef;
use crate::translator::rtyper::lltypesystem::lltype::{
    _ptr, _ptr_obj, LowLevelType, LowLevelValue, Ptr, PtrTarget, cast_int_to_ptr, nullptr,
};
use crate::translator::rtyper::rtyper::RPythonTyper;

/// RPython `class TaggedInstanceRepr(InstanceRepr)` constructor state.
///
/// The Rust struct is a parity shell until `rclass::getinstancerepr`
/// can cache trait-object instance reprs. It preserves the fields that
/// upstream `__init__` adds before `_setup_repr`:
/// `unboxedclassdef` and `is_parent`.
#[derive(Debug)]
pub struct TaggedInstanceRepr {
    rtyper: Weak<RPythonTyper>,
    classdef: Rc<RefCell<ClassDef>>,
    unboxedclassdef: Rc<RefCell<ClassDef>>,
    is_parent: bool,
}

impl TaggedInstanceRepr {
    /// RPython `TaggedInstanceRepr.__init__(rtyper, classdef,
    /// unboxedclassdef)`.
    pub fn new(
        rtyper: &Rc<RPythonTyper>,
        classdef: Rc<RefCell<ClassDef>>,
        unboxedclassdef: Rc<RefCell<ClassDef>>,
    ) -> Self {
        let is_parent = !Rc::ptr_eq(&classdef, &unboxedclassdef);
        TaggedInstanceRepr {
            rtyper: Rc::downgrade(rtyper),
            classdef,
            unboxedclassdef,
            is_parent,
        }
    }

    pub fn rtyper(&self) -> Option<Rc<RPythonTyper>> {
        self.rtyper.upgrade()
    }

    pub fn classdef(&self) -> Rc<RefCell<ClassDef>> {
        self.classdef.clone()
    }

    pub fn unboxedclassdef(&self) -> Rc<RefCell<ClassDef>> {
        self.unboxedclassdef.clone()
    }

    pub fn is_parent(&self) -> bool {
        self.is_parent
    }
}

/// RPython `ll_int_to_unboxed(PTRTYPE, value)`.
///
/// Upstream computes `value * 2 + 1` and then calls
/// `lltype.cast_int_to_ptr(PTRTYPE, oddint)`. The checked arithmetic
/// mirrors the overflow-sensitive class-call lowering in
/// `TaggedInstanceRepr.new_instance`.
pub fn ll_int_to_unboxed(ptrtype: &Ptr, value: i64) -> Result<_ptr, String> {
    let doubled = value
        .checked_mul(2)
        .ok_or_else(|| "ll_int_to_unboxed: value * 2 overflowed".to_string())?;
    let tagged = doubled
        .checked_add(1)
        .ok_or_else(|| "ll_int_to_unboxed: value * 2 + 1 overflowed".to_string())?;
    cast_int_to_ptr(ptrtype, tagged)
}

/// RPython `ll_unboxed_to_int(p)`.
pub fn ll_unboxed_to_int(p: &_ptr) -> Result<i64, String> {
    match p
        ._obj0_value()
        .map_err(|_| "ll_unboxed_to_int: delayed pointer".to_string())?
    {
        Some(_ptr_obj::IntCast(n)) => Ok(n >> 1),
        Some(other) => Err(format!(
            "ll_unboxed_to_int: expected tagged-int pointer, got {other:?}"
        )),
        None => Err("ll_unboxed_to_int: null pointer".to_string()),
    }
}

/// Predicate for upstream's repeated `lltype.cast_ptr_to_int(p) & 1`
/// checks.
pub fn is_unboxed_instance(p: &_ptr) -> Result<bool, String> {
    match p
        ._obj0_value()
        .map_err(|_| "is_unboxed_instance: delayed pointer".to_string())?
    {
        Some(_ptr_obj::IntCast(n)) => Ok((n & 1) != 0),
        Some(_) | None => Ok(false),
    }
}

/// RPython `ll_unboxed_getclass(instance, class_if_unboxed)`.
pub fn ll_unboxed_getclass(instance: &_ptr, class_if_unboxed: &_ptr) -> Result<_ptr, String> {
    if is_unboxed_instance(instance)? {
        return Ok(class_if_unboxed.clone());
    }
    match instance.getattr("typeptr")? {
        LowLevelValue::Ptr(typeptr) => Ok(*typeptr),
        other => Err(format!(
            "ll_unboxed_getclass: instance.typeptr expected Ptr, got {other:?}"
        )),
    }
}

/// RPython `ll_unboxed_getclass_canbenone(instance, class_if_unboxed)`.
pub fn ll_unboxed_getclass_canbenone(
    instance: &_ptr,
    class_if_unboxed: &_ptr,
) -> Result<_ptr, String> {
    if instance.nonzero() {
        return ll_unboxed_getclass(instance, class_if_unboxed);
    }
    let typeptr_type = instance_typeptr_lltype(instance)?;
    nullptr(typeptr_type).map_err(|err| format!("ll_unboxed_getclass_canbenone: {err}"))
}

fn ptr_target_to_lowlevel(target: PtrTarget) -> Result<LowLevelType, String> {
    match target {
        PtrTarget::Struct(st) => Ok(LowLevelType::Struct(Box::new(st))),
        PtrTarget::Array(arr) => Ok(LowLevelType::Array(Box::new(arr))),
        PtrTarget::FixedSizeArray(arr) => Ok(LowLevelType::FixedSizeArray(Box::new(arr))),
        PtrTarget::Opaque(op) => Ok(LowLevelType::Opaque(Box::new(op))),
        PtrTarget::ForwardReference(fwd) => Ok(LowLevelType::ForwardReference(Box::new(fwd))),
        PtrTarget::Func(_) => Err("function pointer target is not a container lltype".to_string()),
    }
}

fn instance_struct_lltype(instance: &_ptr) -> Result<LowLevelType, String> {
    let target = ptr_target_to_lowlevel(instance._TYPE.TO.clone())?;
    match target {
        LowLevelType::ForwardReference(fwd) => fwd.resolved().ok_or_else(|| {
            "ll_unboxed_getclass_canbenone: instance ForwardReference unresolved".to_string()
        }),
        other => Ok(other),
    }
}

fn instance_typeptr_lltype(instance: &_ptr) -> Result<LowLevelType, String> {
    let LowLevelType::Struct(st) = instance_struct_lltype(instance)? else {
        return Err("instance pointer target must be a Struct".to_string());
    };
    let Some(LowLevelType::Ptr(typeptr)) = st._flds.get("typeptr") else {
        return Err("instance Struct has no Ptr typeptr field".to_string());
    };
    ptr_target_to_lowlevel(typeptr.TO.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator::rtyper::lltypesystem::lltype::{
        ForwardReference, LowLevelType, PtrTarget, StructType, nullptr,
    };

    fn gc_ptr_type(name: &str) -> Ptr {
        let fwd = ForwardReference::gc();
        fwd.r#become(LowLevelType::Struct(Box::new(StructType::gc(
            name,
            Vec::new(),
        ))))
        .unwrap();
        Ptr {
            TO: PtrTarget::ForwardReference(fwd),
        }
    }

    #[test]
    fn ll_int_to_unboxed_round_trips_signed_payload() {
        let ptrtype = gc_ptr_type("tagged");
        let p = ll_int_to_unboxed(&ptrtype, -42).expect("tagged ptr");
        assert!(is_unboxed_instance(&p).unwrap());
        assert_eq!(ll_unboxed_to_int(&p).unwrap(), -42);
    }

    #[test]
    fn ll_int_to_unboxed_rejects_overflow_before_cast() {
        let ptrtype = gc_ptr_type("tagged");
        let err = ll_int_to_unboxed(&ptrtype, i64::MAX).unwrap_err();
        assert!(err.contains("overflowed"));
    }

    #[test]
    fn tagged_instance_repr_records_unboxed_relationship() {
        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let parent = ClassDef::new_standalone("pkg.A", None);
        let child = ClassDef::new_standalone("pkg.C", Some(&parent));

        let repr = TaggedInstanceRepr::new(&rtyper, parent.clone(), child.clone());
        assert!(repr.rtyper().is_some());
        assert!(repr.is_parent());
        assert!(Rc::ptr_eq(&repr.classdef(), &parent));
        assert!(Rc::ptr_eq(&repr.unboxedclassdef(), &child));
    }

    #[test]
    fn is_unboxed_instance_is_false_for_null_pointer() {
        let ptrtype = gc_ptr_type("tagged");
        let null = nullptr(LowLevelType::ForwardReference(Box::new(match ptrtype.TO {
            PtrTarget::ForwardReference(fwd) => fwd,
            _ => unreachable!(),
        })))
        .unwrap();
        assert!(!is_unboxed_instance(&null).unwrap());
    }
}
