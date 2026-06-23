//! RPython `rpython/rtyper/rptr.py` parity module.
//!
//! The pointer repr implementations landed in [`super::rmodel`] with
//! upstream `rptr.py` line references.  This module preserves the PyPy
//! import/file path so ports can refer to `rtyper::rptr::PtrRepr` and
//! related symbols by their original RPython home.

pub use crate::annotator::model::SomePtr;
pub use crate::flowspace::model as flowmodel;
pub use crate::translator::rtyper::llannotation::{
    SomeInteriorPtr, SomeLLADTMeth, lltype_to_annotation,
};
pub use crate::translator::rtyper::lltypesystem::lltype;
pub use crate::translator::rtyper::rint::IntegerRepr;
pub use crate::translator::rtyper::rmodel::{InteriorPtrRepr, LLADTMethRepr, PtrRepr, Repr};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rptr_module_exposes_upstream_pointer_repr_symbols() {
        fn assert_repr<T: Repr>() {}

        assert_repr::<PtrRepr>();
        assert_repr::<InteriorPtrRepr>();
        assert_repr::<LLADTMethRepr>();

        let _ptr_new: fn(lltype::Ptr) -> PtrRepr = PtrRepr::new;
        let _interior_new: fn(lltype::InteriorPtr) -> InteriorPtrRepr = InteriorPtrRepr::new;
        let _lltype_to_annotation: fn(lltype::LowLevelType) -> crate::annotator::model::SomeValue =
            lltype_to_annotation::<lltype::LowLevelType>;

        assert_eq!(
            std::any::type_name::<SomePtr>().rsplit("::").next(),
            Some("SomePtr")
        );
        assert_eq!(
            std::any::type_name::<SomeInteriorPtr>().rsplit("::").next(),
            Some("SomeInteriorPtr")
        );
        assert_eq!(
            std::any::type_name::<SomeLLADTMeth>().rsplit("::").next(),
            Some("SomeLLADTMeth")
        );
        assert_eq!(
            std::any::type_name::<IntegerRepr>().rsplit("::").next(),
            Some("IntegerRepr")
        );
    }
}
