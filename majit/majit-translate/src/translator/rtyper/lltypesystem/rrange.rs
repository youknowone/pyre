//! RPython `rpython/rtyper/lltypesystem/rrange.py` parity module.
//!
//! The concrete low-level range repr slice lives in
//! [`crate::translator::rtyper::rrange`] together with the abstract
//! `rpython/rtyper/rrange.py` pieces.  This module keeps the upstream
//! lltypesystem import path available.

pub use crate::translator::rtyper::lltypesystem::lltype;
pub use crate::translator::rtyper::rrange::RangeRepr;

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
}
