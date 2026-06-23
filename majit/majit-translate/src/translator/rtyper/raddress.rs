//! RPython `rpython/rtyper/raddress.py` parity module.
//!
//! The address repr implementations live in [`super::rmodel`] and the
//! low-level address carrier in [`super::lltypesystem::llmemory`].  This
//! module preserves the upstream import/file path for ports that expect
//! `rtyper::raddress::*`.

pub use crate::translator::rtyper::lltypesystem::llmemory::{
    AddressOffset, SomeAddress, SomeTypedAddressAccess, cast_adr_to_ptr, cast_int_to_adr, sizeof,
};
pub use crate::translator::rtyper::lltypesystem::lltype;
pub use crate::translator::rtyper::rint::IntegerRepr;
pub use crate::translator::rtyper::rmodel::{
    AddressRepr, Repr, TypedAddressAccessRepr, address_repr,
};
pub use crate::translator::rtyper::rptr::PtrRepr;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raddress_module_exposes_upstream_address_repr_symbols() {
        fn assert_repr<T: Repr>() {}

        assert_repr::<AddressRepr>();
        assert_repr::<TypedAddressAccessRepr>();

        let repr = address_repr();
        assert_eq!(repr.lowleveltype(), &lltype::LowLevelType::Address);

        let _typed_new: fn(lltype::LowLevelType) -> TypedAddressAccessRepr =
            TypedAddressAccessRepr::new;
        let _cast_int_to_adr: fn(i64) -> Option<lltype::_address> = cast_int_to_adr;

        assert_eq!(
            std::any::type_name::<SomeAddress>().rsplit("::").next(),
            Some("SomeAddress")
        );
        assert_eq!(
            std::any::type_name::<SomeTypedAddressAccess>()
                .rsplit("::")
                .next(),
            Some("SomeTypedAddressAccess")
        );
        assert_eq!(
            std::any::type_name::<IntegerRepr>().rsplit("::").next(),
            Some("IntegerRepr")
        );
        assert_eq!(
            std::any::type_name::<PtrRepr>().rsplit("::").next(),
            Some("PtrRepr")
        );
    }
}
