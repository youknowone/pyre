//! rlocale — RPython: rpython/rlib/rlocale.py
//!
//! `numeric_formatting` is the entry point the number formatter draws its
//! locale from (`newformat.py:643-644`).  It sits beside the `_locale` module
//! port so it shares the raw `localeconv()` walk with that module's own
//! `localeconv()`: the grouping `format(x, 'n')` groups by and the grouping
//! `locale.localeconv()` reports come out of the same read and cannot drift
//! apart.

/// `rlocale.py numeric_formatting`: the decimal point, thousands
/// separator and grouping string of the current locale, as the bytes
/// `localeconv()` reports them.
///
/// Without `host_env` and under sandbox the C locale's values stand in.
/// Upstream declares `localeconv` `sandboxsafe=True` (`rlocale.py`,
/// `:180-182`) and reads the host locale even there; pyre compiles the call out
/// instead, because the sandbox build replaces `_locale`'s host entry points
/// with raising stubs and `format()` must not acquire a raising path.
pub fn numeric_formatting() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    #[cfg(all(any(unix, windows), feature = "host_env", not(feature = "sandbox")))]
    {
        return rustpython_host_env::locale::localeconv_numeric();
    }
    #[cfg(not(all(any(unix, windows), feature = "host_env", not(feature = "sandbox"))))]
    (b".".to_vec(), Vec::new(), Vec::new())
}
