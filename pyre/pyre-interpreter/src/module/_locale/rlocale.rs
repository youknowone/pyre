//! rlocale — RPython: rpython/rlib/rlocale.py
//!
//! `numeric_formatting` is the entry point the number formatter draws its
//! locale from (`newformat.py:643-644`).  It sits beside the `_locale` module
//! port so it shares the raw `localeconv()` walk with that module's own
//! `localeconv()`: the grouping `format(x, 'n')` groups by and the grouping
//! `locale.localeconv()` reports come out of the same read and cannot drift
//! apart.

/// Every byte of a NUL-terminated C string, `CHAR_MAX` included.
///
/// `rffi.charp2str` (`rlocale.py:175-177`) truncates at the NUL and at nothing
/// else, so a grouping terminator a locale spells as `CHAR_MAX` stays in the
/// result.  `rustpython_host_env::locale`'s own reader stops at `CHAR_MAX` and
/// drops it, which would collapse the "stop" and "repeat the last group"
/// conventions onto the same vector.
#[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
pub(super) fn charp2str(ptr: *const libc::c_char) -> Vec<u8> {
    let mut out = Vec::new();
    if !ptr.is_null() {
        let mut cur = ptr;
        unsafe {
            while *cur != 0 {
                out.push(*cur as u8);
                cur = cur.add(1);
            }
        }
    }
    out
}

/// `rlocale.py:173-178 numeric_formatting`: the decimal point, thousands
/// separator and grouping string of the current locale, as the bytes
/// `localeconv()` reports them.
///
/// Off unix, without `host_env`, and under sandbox the C locale's values stand
/// in.  Upstream declares `localeconv` `sandboxsafe=True` (`rlocale.py:167`,
/// `:180-182`) and reads the host locale even there; pyre compiles the call out
/// instead, because the sandbox build replaces `_locale`'s host entry points
/// with raising stubs and `format()` must not acquire a raising path.
pub(crate) fn numeric_formatting() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    {
        let conv = unsafe { libc::localeconv() };
        if !conv.is_null() {
            return unsafe {
                (
                    charp2str((*conv).decimal_point),
                    charp2str((*conv).thousands_sep),
                    charp2str((*conv).grouping),
                )
            };
        }
    }
    (b".".to_vec(), Vec::new(), Vec::new())
}
