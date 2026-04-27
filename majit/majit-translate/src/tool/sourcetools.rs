//! Port of `rpython/tool/sourcetools.py`.
//!
//! Only the subset of helpers used by downstream ports lands; the
//! upstream module also carries `func_with_new_name`,
//! `nice_repr_for_func`, `has_varargs`, etc., none of which currently
//! have a Rust caller.

/// Maximum length of a `valid_identifier` result, mirroring upstream
/// `sourcetools.py:239 PY_IDENTIFIER_MAX = 120`.
pub const PY_IDENTIFIER_MAX: usize = 120;

/// RPython `rpython.tool.sourcetools.valid_identifier`
/// (`sourcetools.py:241-245`).
///
/// Translate `stuff` through `PY_IDENTIFIER` (every byte that is not
/// `[0-9A-Za-z]` becomes `_`), prepend `_` if the first byte is
/// otherwise a digit (or empty), then truncate to
/// `PY_IDENTIFIER_MAX`.
pub fn valid_identifier(stuff: impl std::fmt::Display) -> String {
    let mut stuff: String = stuff
        .to_string()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect();
    if stuff.is_empty() || stuff.as_bytes()[0].is_ascii_digit() {
        stuff.insert(0, '_');
    }
    stuff.truncate(PY_IDENTIFIER_MAX);
    stuff
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alphanumeric_passes_through() {
        assert_eq!(valid_identifier("abc123"), "abc123");
    }

    #[test]
    fn non_alpha_chars_become_underscores() {
        assert_eq!(valid_identifier("a-b.c d"), "a_b_c_d");
    }

    #[test]
    fn leading_digit_is_prefixed_with_underscore() {
        assert_eq!(valid_identifier("9foo"), "_9foo");
    }

    #[test]
    fn empty_string_becomes_underscore() {
        assert_eq!(valid_identifier(""), "_");
    }

    #[test]
    fn long_input_truncates_to_max() {
        let s = "x".repeat(200);
        let out = valid_identifier(&s);
        assert_eq!(out.len(), PY_IDENTIFIER_MAX);
        assert!(out.chars().all(|c| c == 'x'));
    }
}
