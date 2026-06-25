//! RPython `rpython/rtyper/rbuilder.py` parity module.
//!
//! Upstream splits string-builder rtyping into this abstract method
//! surface and a concrete lltypesystem implementation in
//! `rpython/rtyper/lltypesystem/rbuilder.py`. The lltypesystem half is
//! still pending in pyre, so this module records the exact abstract
//! `AbstractStringBuilderRepr` method names first. That keeps callers
//! and future ports using the upstream names instead of inventing local
//! aliases.

use crate::translator::rtyper::error::TyperError;

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

/// Method names handled by
/// `AbstractStringBuilderRepr.rtype_method_<name>`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StringBuilderMethod {
    /// `rtype_method_append`
    Append,
    /// `rtype_method_append_slice`
    AppendSlice,
    /// `rtype_method_append_multiple_char`
    AppendMultipleChar,
    /// `rtype_method_append_charpsize`
    AppendCharpsize,
    /// `rtype_method_getlength`
    Getlength,
    /// `rtype_method_build`
    Build,
}

impl StringBuilderMethod {
    /// RPython method suffix used by `BuiltinMethodRepr`.
    pub const fn as_method_name(self) -> &'static str {
        match self {
            StringBuilderMethod::Append => "append",
            StringBuilderMethod::AppendSlice => "append_slice",
            StringBuilderMethod::AppendMultipleChar => "append_multiple_char",
            StringBuilderMethod::AppendCharpsize => "append_charpsize",
            StringBuilderMethod::Getlength => "getlength",
            StringBuilderMethod::Build => "build",
        }
    }

    /// RPython implementation method name on `AbstractStringBuilderRepr`.
    pub const fn as_rtype_method_name(self) -> &'static str {
        match self {
            StringBuilderMethod::Append => "rtype_method_append",
            StringBuilderMethod::AppendSlice => "rtype_method_append_slice",
            StringBuilderMethod::AppendMultipleChar => "rtype_method_append_multiple_char",
            StringBuilderMethod::AppendCharpsize => "rtype_method_append_charpsize",
            StringBuilderMethod::Getlength => "rtype_method_getlength",
            StringBuilderMethod::Build => "rtype_method_build",
        }
    }
}

/// Upstream method table from `rbuilder.py:13-46`.
pub const STRING_BUILDER_METHODS: [StringBuilderMethod; 6] = [
    StringBuilderMethod::Append,
    StringBuilderMethod::AppendSlice,
    StringBuilderMethod::AppendMultipleChar,
    StringBuilderMethod::AppendCharpsize,
    StringBuilderMethod::Getlength,
    StringBuilderMethod::Build,
];

impl AbstractStringBuilderRepr {
    /// Resolve a `BuiltinMethodRepr.methodname` suffix to the upstream
    /// `rtype_method_*` arm. Unknown names raise the same missing-method
    /// shape as the base `Repr.rtype_method` dispatcher.
    pub fn method_from_name(method_name: &str) -> Result<StringBuilderMethod, TyperError> {
        STRING_BUILDER_METHODS
            .iter()
            .copied()
            .find(|method| method.as_method_name() == method_name)
            .ok_or_else(|| {
                TyperError::message(format!(
                    "missing AbstractStringBuilderRepr.rtype_method_{method_name}"
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AbstractStringBuilderRepr, INIT_SIZE, STRING_BUILDER_METHODS, StringBuilderMethod,
    };

    #[test]
    fn init_size_matches_rlib_rstring_default() {
        assert_eq!(INIT_SIZE, 100);
    }

    #[test]
    fn abstract_string_builder_method_names_match_rpython_surface() {
        let names: Vec<_> = STRING_BUILDER_METHODS
            .iter()
            .map(|method| method.as_rtype_method_name())
            .collect();
        assert_eq!(
            names,
            vec![
                "rtype_method_append",
                "rtype_method_append_slice",
                "rtype_method_append_multiple_char",
                "rtype_method_append_charpsize",
                "rtype_method_getlength",
                "rtype_method_build",
            ]
        );
    }

    #[test]
    fn method_from_name_resolves_builtin_method_suffixes() {
        assert_eq!(
            AbstractStringBuilderRepr::method_from_name("append").unwrap(),
            StringBuilderMethod::Append
        );
        assert_eq!(
            AbstractStringBuilderRepr::method_from_name("append_charpsize").unwrap(),
            StringBuilderMethod::AppendCharpsize
        );
        assert!(AbstractStringBuilderRepr::method_from_name("extend").is_err());
    }
}
