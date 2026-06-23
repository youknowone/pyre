//! RPython `rpython/rtyper/debug.py` parity module.
//!
//! The upstream file contains runtime helpers plus `ExtRegistryEntry`
//! specializers that lower `ll_assert` and `ll_assert_not_none` to
//! `debug_assert*` low-level ops.  The op descriptors live in
//! [`crate::translator::rtyper::lltypesystem::lloperation`]; this module
//! keeps the helper names available under the upstream import path.

use std::fmt;

/// RPython `ll_assert(x, msg)` (`debug.py:4-7`).
pub fn ll_assert(x: bool, msg: &str) {
    assert!(x, "{msg}");
}

/// RPython `ll_assert_not_none(x)` (`debug.py:19-22`).
pub fn ll_assert_not_none<T>(x: Option<T>) -> T {
    x.unwrap_or_else(|| panic!("ll_assert_not_none(None)"))
}

/// RPython `class FatalError(Exception)` (`debug.py:42-43`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FatalError {
    pub msg: String,
}

impl FatalError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self { msg: msg.into() }
    }
}

impl fmt::Display for FatalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}

impl std::error::Error for FatalError {}

/// RPython `fatalerror(msg)` (`debug.py:45-53`).
///
/// Pyre runs this helper in the untranslated Rust host, so it follows
/// upstream's `not we_are_translated()` branch and returns `FatalError`.
pub fn fatalerror(msg: impl Into<String>) -> Result<(), FatalError> {
    Err(FatalError::new(msg))
}

/// RPython `fatalerror_notb(msg)` (`debug.py:55-62`).
pub fn fatalerror_notb(msg: impl Into<String>) -> Result<(), FatalError> {
    Err(FatalError::new(msg))
}

/// RPython `debug_print_traceback()` (`debug.py:64-70`).
///
/// The untranslated helper is intentionally side-effect free locally; the
/// translated operation is represented by the `debug_print_traceback` llop.
pub fn debug_print_traceback() {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator::rtyper::lltypesystem::lloperation::ll_operations;

    #[test]
    fn debug_helpers_match_untranslated_runtime_surface() {
        ll_assert(true, "ok");
        assert_eq!(ll_assert_not_none(Some(3)), 3);

        let err = fatalerror("boom").unwrap_err();
        assert_eq!(err.msg, "boom");
        assert_eq!(
            fatalerror_notb("no traceback").unwrap_err().msg,
            "no traceback"
        );
        debug_print_traceback();
    }

    #[test]
    fn debug_module_lines_up_with_low_level_debug_ops() {
        let ops = ll_operations();
        assert!(ops.contains_key("debug_assert"));
        assert!(ops.contains_key("debug_assert_not_none"));
        assert!(ops.contains_key("debug_fatalerror"));
        assert!(ops.contains_key("debug_print_traceback"));
    }
}
