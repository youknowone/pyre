//! RPython `rpython/rtyper/rbuilder.py` parity module.
//!
//! Upstream splits string-builder rtyping into an abstract method
//! surface here and a concrete lltypesystem implementation in
//! `rpython/rtyper/lltypesystem/rbuilder.py`. The lltypesystem half is
//! still pending in pyre, so this module records the upstream
//! `AbstractStringBuilderRepr` name without inventing a local dispatcher
//! shape.

/// RPython `rpython.rlib.rstring.INIT_SIZE`.
///
/// `AbstractStringBuilderRepr.rtyper_new` uses this default when the
/// high-level constructor receives no explicit initial size.
pub const INIT_SIZE: i64 = 100;

/// RPython `class AbstractStringBuilderRepr(Repr)`.
///
/// The concrete low-level fields (`ll_new`, `ll_append`, `ll_build`,
/// etc.) are supplied by `lltypesystem/rbuilder.py` upstream. Pyre
/// exposes this marker before that concrete repr lands so module and
/// class names already line up with RPython.
#[derive(Debug, Default)]
pub struct AbstractStringBuilderRepr;

#[cfg(test)]
mod tests {
    use super::{AbstractStringBuilderRepr, INIT_SIZE};

    #[test]
    fn init_size_matches_rlib_rstring_default() {
        assert_eq!(INIT_SIZE, 100);
    }

    #[test]
    fn abstract_repr_name_is_available() {
        let repr = AbstractStringBuilderRepr;
        assert_eq!(format!("{repr:?}"), "AbstractStringBuilderRepr");
    }
}
