//! RPython `rpython/rtyper/lltypesystem/rrange.py` parity module.
//!
//! The concrete low-level range repr slice lives in
//! [`crate::translator::rtyper::rrange`] together with the abstract
//! `rpython/rtyper/rrange.py` pieces.  This module keeps the upstream
//! lltypesystem import path available.

#![allow(non_snake_case)]

use std::sync::LazyLock;

use crate::translator::rtyper::error::TyperError;

pub use crate::translator::rtyper::lltypesystem::lltype;
use crate::translator::rtyper::lltypesystem::lltype::{
    _ptr, LowLevelType, LowLevelValue, MallocFlavor, Ptr, PtrTarget, Struct, malloc,
};
pub use crate::translator::rtyper::rrange::AbstractRangeRepr as RangeRepr;

/// RPython `ll_length(l)`.
pub fn ll_length(l: &_ptr) -> Result<i64, TyperError> {
    let start = signed_field(l, "start")?;
    let stop = signed_field(l, "stop")?;
    let step = signed_field(l, "step")?;
    let (lo, hi, step) = if step > 0 {
        (start, stop, step)
    } else {
        (stop, start, -step)
    };
    if hi <= lo {
        return Ok(0);
    }
    Ok((hi - lo - 1) / step + 1)
}

/// RPython `ll_getitem_fast(l, index)`.
pub fn ll_getitem_fast(l: &_ptr, index: i64) -> Result<i64, TyperError> {
    Ok(signed_field(l, "start")? + index * signed_field(l, "step")?)
}

fn range_struct(step_field: bool) -> LowLevelType {
    let mut fields = vec![
        ("start".to_string(), LowLevelType::Signed),
        ("stop".to_string(), LowLevelType::Signed),
    ];
    if step_field {
        fields.push(("step".to_string(), LowLevelType::Signed));
    }
    LowLevelType::Struct(Box::new(Struct::gc_with_hints(
        "range",
        fields,
        vec![(
            "immutable".to_string(),
            crate::flowspace::model::ConstValue::Bool(true),
        )],
    )))
}

fn range_iter_struct(step_field: bool) -> LowLevelType {
    let mut fields = vec![
        ("next".to_string(), LowLevelType::Signed),
        ("stop".to_string(), LowLevelType::Signed),
    ];
    if step_field {
        fields.push(("step".to_string(), LowLevelType::Signed));
    }
    LowLevelType::Struct(Box::new(Struct::gc("range", fields)))
}

/// RPython `RANGEST = GcStruct("range", start, stop, step, ...)`.
pub static RANGEST: LazyLock<LowLevelType> = LazyLock::new(|| range_struct(true));

/// RPython `RANGESTITER = GcStruct("range", next, stop, step)`.
pub static RANGESTITER: LazyLock<LowLevelType> = LazyLock::new(|| range_iter_struct(true));

/// RPython `ll_newrange(RANGE, start, stop)`.
pub fn ll_newrange(RANGE: &Ptr, start: i64, stop: i64) -> Result<_ptr, TyperError> {
    let mut l = malloc(RANGE.TO.clone().into(), None, MallocFlavor::Gc, false)
        .map_err(TyperError::message)?;
    l.setattr("start", LowLevelValue::Signed(start))
        .map_err(TyperError::message)?;
    l.setattr("stop", LowLevelValue::Signed(stop))
        .map_err(TyperError::message)?;
    Ok(l)
}

/// RPython `ll_newrangest(start, stop, step)`.
pub fn ll_newrangest(start: i64, stop: i64, step: i64) -> Result<_ptr, TyperError> {
    if step == 0 {
        return Err(TyperError::message("ValueError"));
    }
    let mut l =
        malloc((*RANGEST).clone(), None, MallocFlavor::Gc, false).map_err(TyperError::message)?;
    l.setattr("start", LowLevelValue::Signed(start))
        .map_err(TyperError::message)?;
    l.setattr("stop", LowLevelValue::Signed(stop))
        .map_err(TyperError::message)?;
    l.setattr("step", LowLevelValue::Signed(step))
        .map_err(TyperError::message)?;
    Ok(l)
}

/// RPython `class RangeIteratorRepr(AbstractRangeIteratorRepr)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeIteratorRepr {
    pub lowleveltype: LowLevelType,
}

/// RPython `ll_rangeiter(ITERPTR, rng)`.
pub fn ll_rangeiter(ITERPTR: &Ptr, rng: &_ptr) -> Result<_ptr, TyperError> {
    let mut iter = malloc(ITERPTR.TO.clone().into(), None, MallocFlavor::Gc, false)
        .map_err(TyperError::message)?;
    iter.setattr("next", LowLevelValue::Signed(signed_field(rng, "start")?))
        .map_err(TyperError::message)?;
    iter.setattr("stop", LowLevelValue::Signed(signed_field(rng, "stop")?))
        .map_err(TyperError::message)?;
    let is_rangestiter = match &ITERPTR.TO {
        PtrTarget::Struct(st) => *st == ptr_target_struct(&RANGESTITER)?,
        _ => false,
    };
    if is_rangestiter {
        iter.setattr("step", LowLevelValue::Signed(signed_field(rng, "step")?))
            .map_err(TyperError::message)?;
    }
    Ok(iter)
}

fn ptr_target_struct(ty: &LowLevelType) -> Result<Struct, TyperError> {
    match ty {
        LowLevelType::Struct(st) => Ok((**st).clone()),
        other => Err(TyperError::message(format!(
            "expected Struct, got {other:?}"
        ))),
    }
}

fn signed_field(ptr: &_ptr, field: &str) -> Result<i64, TyperError> {
    match ptr.getattr(field).map_err(TyperError::message)? {
        LowLevelValue::Signed(n) => Ok(n),
        other => Err(TyperError::message(format!(
            "rrange.{field}: expected Signed, got {other:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator::rtyper::rmodel::Repr;

    #[test]
    fn lltypesystem_rrange_exposes_concrete_range_repr_path() {
        let repr = RangeRepr::new(1).expect("step-1 RangeRepr");
        assert_eq!(repr.class_name(), "RangeRepr");
        assert!(matches!(repr.lowleveltype(), lltype::LowLevelType::Ptr(_)));
    }

    #[test]
    fn lltypesystem_rrange_helpers_allocate_and_iterate_like_upstream() {
        let range_ptr = match RangeRepr::new(1).expect("range repr").lowleveltype() {
            lltype::LowLevelType::Ptr(ptr) => ptr.clone(),
            other => panic!("RangeRepr lowleveltype must be Ptr, got {other:?}"),
        };
        let rng = ll_newrange(&range_ptr, 3, 8).expect("ll_newrange");
        assert_eq!(signed_field(&rng, "start").expect("start"), 3);
        assert_eq!(signed_field(&rng, "stop").expect("stop"), 8);

        let rangest = ll_newrangest(10, 2, -2).expect("ll_newrangest");
        assert_eq!(ll_length(&rangest).expect("ll_length"), 4);
        assert_eq!(ll_getitem_fast(&rangest, 2).expect("getitem"), 6);
        assert!(ll_newrangest(0, 1, 0).is_err());

        let iter_type = match &*RANGESTITER {
            lltype::LowLevelType::Struct(st) => Ptr {
                TO: PtrTarget::Struct((**st).clone()),
            },
            other => panic!("RANGESTITER must be Struct, got {other:?}"),
        };
        let iter = ll_rangeiter(&iter_type, &rangest).expect("ll_rangeiter");
        assert_eq!(signed_field(&iter, "next").expect("next"), 10);
        assert_eq!(signed_field(&iter, "stop").expect("stop"), 2);
        assert_eq!(signed_field(&iter, "step").expect("step"), -2);

        let iter_repr = RangeIteratorRepr {
            lowleveltype: lltype::LowLevelType::Ptr(Box::new(iter_type)),
        };
        assert!(matches!(
            iter_repr.lowleveltype,
            lltype::LowLevelType::Ptr(_)
        ));
    }
}
