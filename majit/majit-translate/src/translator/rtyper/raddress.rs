//! RPython `rpython/rtyper/raddress.py` parity module.
//!
//! The address repr implementations live in [`super::rmodel`] and the
//! low-level address carrier in [`super::lltypesystem::llmemory`].  This
//! module preserves the upstream import/file path for ports that expect
//! `rtyper::raddress::*`.

pub use crate::translator::rtyper::lltypesystem::llmemory::{
    AddressOffset, SomeAddress, SomeTypedAddressAccess, cast_adr_to_int, cast_adr_to_ptr,
    cast_int_to_adr, sizeof,
};
pub use crate::translator::rtyper::lltypesystem::lltype;
pub use crate::translator::rtyper::rint::IntegerRepr;
pub use crate::translator::rtyper::rmodel::{
    AddressRepr, Repr, TypedAddressAccessRepr, address_repr,
};
pub use crate::translator::rtyper::rptr::PtrRepr;

/// RPython `Address` / `fakeaddress` import surface from
/// `lltypesystem.llmemory` (`raddress.py:5-7`).
pub type Address = lltype::_address;

#[allow(non_camel_case_types)]
pub type fakeaddress = lltype::_address;

/// RPython `NULL = fakeaddress(None)` import surface (`raddress.py:5-7`).
pub const NULL: Address = lltype::_address::Null;

/// RPython `ll_addrhash(addr1)` (`raddress.py:64-65`).
pub fn ll_addrhash(addr1: &Address) -> Result<i64, String> {
    cast_adr_to_int(addr1, Some("forced"))
}

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
        let _cast_adr_to_int: fn(&lltype::_address, Option<&str>) -> Result<i64, String> =
            cast_adr_to_int;

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

    #[test]
    fn raddress_module_exposes_address_constants_and_hash_helper() {
        let null: Address = NULL;
        let fake: fakeaddress = lltype::_address::IntCast(41);

        assert_eq!(null, lltype::_address::Null);
        assert_eq!(ll_addrhash(&null), Ok(0));
        assert_eq!(ll_addrhash(&fake), Ok(41));
    }
}
