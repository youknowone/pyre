//! RPython `rpython/rtyper/lltypesystem/rlist.py` parity module.
//!
//! The list repr slice lives in [`crate::translator::rtyper::rlist`]
//! together with the abstract `rpython/rtyper/rlist.py` pieces.  This
//! module keeps the upstream lltypesystem import path available.

pub use crate::translator::rtyper::lltypesystem::lltype;
pub use crate::translator::rtyper::rlist::{FixedSizeListRepr, ListRepr};
pub use crate::translator::rtyper::rmodel::Repr;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lltypesystem_rlist_exposes_concrete_list_repr_paths() {
        fn assert_repr<T: Repr>() {}

        assert_repr::<FixedSizeListRepr>();
        assert_repr::<ListRepr>();

        assert_eq!(
            std::any::type_name::<FixedSizeListRepr>()
                .rsplit("::")
                .next(),
            Some("FixedSizeListRepr")
        );
        assert_eq!(
            std::any::type_name::<ListRepr>().rsplit("::").next(),
            Some("ListRepr")
        );
    }
}
