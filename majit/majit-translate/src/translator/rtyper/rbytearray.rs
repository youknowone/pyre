//! RPython `rpython/rtyper/rbytearray.py` parity module.
//!
//! The concrete lltypesystem representation lives in
//! [`crate::translator::rtyper::lltypesystem::rbytearray`], matching
//! upstream's split between `rbytearray.py` and
//! `lltypesystem/rbytearray.py`.

pub use crate::translator::rtyper::lltypesystem::rbytearray::{ByteArrayRepr, bytearray_repr};

/// RPython `class AbstractByteArrayRepr(AbstractStringRepr)`.
///
/// The Rust port records the inheritance in
/// [`crate::translator::rtyper::pairtype::ReprClassId::ByteArrayRepr`]'s
/// MRO. This marker keeps the top-level module's named surface aligned
/// with upstream.
#[derive(Debug, Default)]
pub struct AbstractByteArrayRepr;

#[cfg(test)]
mod tests {
    use crate::translator::rtyper::pairtype::{ReprClassId, pair_mro};
    use crate::translator::rtyper::rmodel::Repr;

    #[test]
    fn bytearray_repr_inherits_abstract_string_repr_in_pair_mro() {
        let repr = super::bytearray_repr();
        assert_eq!(repr.class_name(), "ByteArrayRepr");
        assert_eq!(repr.repr_class_id(), ReprClassId::ByteArrayRepr);
        assert!(
            pair_mro(ReprClassId::ByteArrayRepr, ReprClassId::StringRepr).contains(&(
                ReprClassId::AbstractStringRepr,
                ReprClassId::AbstractStringRepr,
            ))
        );
    }
}
