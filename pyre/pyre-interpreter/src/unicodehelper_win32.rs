//! `unicodehelper_win32.py` — the Windows code page codecs.
//!
//! `MultiByteToWideChar` / `WideCharToMultiByte` are reached directly rather
//! than through a mapping table: the table is the operating system's, and
//! which one applies is chosen per call by code page number.  `mbcs`, `oem`
//! and every `cpNNN` that `encodings/__init__.py`'s
//! `win32_code_page_search_function` builds land here.

use pyre_object::{PY_NULL, PyObjectRef};
use rustpython_wtf8::{Wtf8, Wtf8Buf};
use windows_sys::Win32::Foundation::{
    ERROR_INSUFFICIENT_BUFFER, ERROR_INVALID_FLAGS, ERROR_NO_UNICODE_TRANSLATION,
};
use windows_sys::Win32::Globalization::{
    CP_ACP, CP_UTF7, CP_UTF8, MB_ERR_INVALID_CHARS, MultiByteToWideChar, WC_ERR_INVALID_CHARS,
    WC_NO_BEST_FIT_CHARS, WideCharToMultiByte,
};

/// The Windows 2000 English message the decoder reports, which
/// `decode_code_page_errors` hardcodes rather than asking `FormatMessage` for.
const DECODE_REASON: &str = "No mapping for the Unicode character exists in the target code page.";

/// The encoder's counterpart, likewise fixed rather than looked up.
const ENCODE_REASON: &str = "invalid character";

/// A code page's codec name: the active ANSI code page is `mbcs`, everything
/// else is its own number.  `CP_OEMCP` is 1 rather than the resolved OEM code
/// page, so `oem` reports `cp1` — the number that was asked for, not the one
/// the system chose for it.
pub fn code_page_name(code_page: u32) -> String {
    if code_page == CP_ACP {
        "mbcs".to_owned()
    } else {
        format!("cp{code_page}")
    }
}

/// The CP_UTF7 decoder accepts no flags at all; every other code page takes
/// `MB_ERR_INVALID_CHARS`, so an unmappable byte fails rather than becoming
/// U+FFFD behind the error handler's back.
fn decode_code_page_flags(code_page: u32) -> u32 {
    if code_page == CP_UTF7 {
        0
    } else {
        MB_ERR_INVALID_CHARS
    }
}

/// `WC_NO_BEST_FIT_CHARS` is what makes an unencodable character report itself
/// through `lpUsedDefaultChar` instead of silently becoming a lookalike.
/// `replace` drops it: that caller has already asked for a substitution.
fn encode_code_page_flags(code_page: u32, errors: Option<&str>) -> u32 {
    if code_page == CP_UTF8 {
        WC_ERR_INVALID_CHARS
    } else if code_page == CP_UTF7 {
        0
    } else if errors == Some("replace") {
        0
    } else {
        WC_NO_BEST_FIT_CHARS
    }
}

fn last_win32_error() -> u32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0) as u32
}

fn win32_error() -> crate::PyError {
    crate::PyError::os_error_win32_syscall2(last_win32_error() as i32, PY_NULL, PY_NULL)
}

/// The outcome of a conversion the caller retries through an error handler.
enum CodePageFail {
    /// `ERROR_NO_UNICODE_TRANSLATION` — the input has no spelling in the
    /// target, which is the error handler's business rather than the system's.
    NoTranslation,
    /// Anything else the system reported, which is reported as it stands.
    Os(crate::PyError),
}

/// `decode_code_page_strict` — the whole buffer in one call, which is the
/// answer whenever every byte maps.
fn decode_strict(code_page: u32, data: &[u8]) -> Result<Vec<u16>, CodePageFail> {
    debug_assert!(!data.is_empty());
    let mut flags = decode_code_page_flags(code_page);
    let insize = data.len() as i32;
    let outsize = loop {
        let size = unsafe {
            MultiByteToWideChar(
                code_page,
                flags,
                data.as_ptr(),
                insize,
                std::ptr::null_mut(),
                0,
            )
        };
        if size > 0 {
            break size;
        }
        // Some code pages — UTF-7 among them — reject any flag word at all.
        if flags != 0 && last_win32_error() == ERROR_INVALID_FLAGS {
            flags = 0;
            continue;
        }
        return Err(decode_fail());
    };
    let mut out = vec![0u16; outsize as usize];
    let written = unsafe {
        MultiByteToWideChar(
            code_page,
            flags,
            data.as_ptr(),
            insize,
            out.as_mut_ptr(),
            outsize,
        )
    };
    if written <= 0 {
        return Err(decode_fail());
    }
    out.truncate(written as usize);
    Ok(out)
}

/// Classify the failure a conversion call just reported.
fn decode_fail() -> CodePageFail {
    if last_win32_error() == ERROR_NO_UNICODE_TRANSLATION {
        CodePageFail::NoTranslation
    } else {
        CodePageFail::Os(win32_error())
    }
}

/// `decode_code_page_errors` — one character at a time, so the error handler
/// sees the byte that failed rather than the whole buffer.
fn decode_errors(
    code_page: u32,
    data: &[u8],
    errors: &str,
    is_final: bool,
) -> Result<(Wtf8Buf, usize), crate::PyError> {
    let encoding = code_page_name(code_page);
    // A strict decode of a complete buffer has nothing left to try, so the
    // whole input is what gets reported — not the byte a walk would reach.
    if errors == "strict" && is_final {
        return Err(crate::typedef::unicode_decode_error(
            &encoding,
            data,
            0,
            0,
            DECODE_REASON,
        ));
    }
    let mut flags = decode_code_page_flags(code_page);
    let mut out = Wtf8Buf::new();
    let mut pos = 0usize;
    while pos < data.len() {
        // A character is at most 4 bytes and decodes to at most a surrogate
        // pair, so the input span grows until one width converts.
        let mut insize = 1usize;
        let mut wide = [0u16; 2];
        let converted = loop {
            let size = unsafe {
                MultiByteToWideChar(
                    code_page,
                    flags,
                    data[pos..].as_ptr(),
                    insize as i32,
                    wide.as_mut_ptr(),
                    wide.len() as i32,
                )
            };
            if size > 0 {
                break size as usize;
            }
            let err = last_win32_error();
            if err == ERROR_INVALID_FLAGS && flags != 0 {
                flags = 0;
                continue;
            }
            if err != ERROR_NO_UNICODE_TRANSLATION && err != ERROR_INSUFFICIENT_BUFFER {
                return Err(win32_error());
            }
            insize += 1;
            if insize > 4 || pos + insize > data.len() {
                break 0;
            }
        };
        if converted > 0 {
            out.push_wtf8(&Wtf8Buf::from_wide(&wide[..converted]));
            pos += insize;
            continue;
        }
        // The span that failed runs to the end of the buffer, so more bytes
        // could still complete it: stop and leave them to the caller.
        if pos + insize >= data.len() && !is_final {
            break;
        }
        let (newpos, _) = crate::type_methods::call_registered_decode_error_handler(
            errors,
            &encoding,
            data,
            pos,
            pos + 1,
            DECODE_REASON,
            &mut out,
        )?;
        pos = newpos;
    }
    Ok((out, pos))
}

/// `PyUnicode_DecodeCodePageStateful` — the decoded text and how many bytes of
/// `data` it consumed.  `is_final` says no more bytes are coming, which is what
/// turns a truncated trailing sequence from "wait" into an error.
pub fn decode_code_page(
    code_page: u32,
    data: &[u8],
    errors: &str,
    is_final: bool,
) -> Result<(Wtf8Buf, usize), crate::PyError> {
    if data.is_empty() {
        return Ok((Wtf8Buf::new(), 0));
    }
    match decode_strict(code_page, data) {
        Ok(wide) => Ok((Wtf8Buf::from_wide(&wide), data.len())),
        Err(CodePageFail::NoTranslation) => decode_errors(code_page, data, errors, is_final),
        Err(CodePageFail::Os(error)) => Err(error),
    }
}

/// Whether this code page reports an unencodable character through
/// `lpUsedDefaultChar`.  The two Unicode transformations take no such
/// argument, since every character has a spelling in them.
fn uses_default_char(code_page: u32) -> bool {
    code_page != CP_UTF8 && code_page != CP_UTF7
}

/// `encode_code_page_strict` — the whole string in one call, which is the
/// answer whenever every character maps.
fn encode_strict(code_page: u32, wide: &[u16]) -> Result<Vec<u8>, CodePageFail> {
    debug_assert!(!wide.is_empty());
    let flags = encode_code_page_flags(code_page, None);
    let mut used_default = 0i32;
    let used_default_ptr = if uses_default_char(code_page) {
        &raw mut used_default
    } else {
        std::ptr::null_mut()
    };
    let insize = wide.len() as i32;
    let outsize = unsafe {
        WideCharToMultiByte(
            code_page,
            flags,
            wide.as_ptr(),
            insize,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            used_default_ptr,
        )
    };
    if outsize <= 0 {
        return Err(encode_fail());
    }
    // A default character stands in for one this code page cannot spell, so
    // its use is the failure the per-character walk is there to report.
    if used_default != 0 {
        return Err(CodePageFail::NoTranslation);
    }
    let mut out = vec![0u8; outsize as usize];
    let written = unsafe {
        WideCharToMultiByte(
            code_page,
            flags,
            wide.as_ptr(),
            insize,
            out.as_mut_ptr(),
            outsize,
            std::ptr::null(),
            used_default_ptr,
        )
    };
    if written <= 0 {
        return Err(encode_fail());
    }
    if used_default != 0 {
        return Err(CodePageFail::NoTranslation);
    }
    out.truncate(written as usize);
    Ok(out)
}

/// [`decode_fail`]'s counterpart for the encoding direction.
fn encode_fail() -> CodePageFail {
    if last_win32_error() == ERROR_NO_UNICODE_TRANSLATION {
        CodePageFail::NoTranslation
    } else {
        CodePageFail::Os(win32_error())
    }
}

/// One code point through `WideCharToMultiByte`, appended to `out`.
///
/// `Ok(false)` is the code page having no spelling for it, which is the error
/// handler's business; `Err` is the system reporting anything else.  The walk
/// is over code points rather than the code units behind them, so an astral
/// character fails once instead of once per surrogate.
fn encode_one(
    code_page: u32,
    flags: u32,
    takes_default: bool,
    ch: u32,
    out: &mut Vec<u8>,
) -> Result<bool, crate::PyError> {
    let mut chars = [0u16; 2];
    let charsize = if ch < 0x10000 {
        chars[0] = ch as u16;
        1
    } else {
        chars[0] = (0xD800 - (0x10000 >> 10) + (ch >> 10)) as u16;
        chars[1] = (0xDC00 + (ch & 0x3FF)) as u16;
        2
    };
    let mut used_default = 0i32;
    let used_default_ptr = if takes_default {
        &raw mut used_default
    } else {
        std::ptr::null_mut()
    };
    // 4 is the longest sequence any code page spells one character in.
    let mut buffer = [0u8; 4];
    let outsize = unsafe {
        WideCharToMultiByte(
            code_page,
            flags,
            chars.as_ptr(),
            charsize,
            buffer.as_mut_ptr(),
            buffer.len() as i32,
            std::ptr::null(),
            used_default_ptr,
        )
    };
    if outsize > 0 {
        if used_default == 0 {
            out.extend_from_slice(&buffer[..outsize as usize]);
            return Ok(true);
        }
    } else if last_win32_error() != ERROR_NO_UNICODE_TRANSLATION {
        return Err(win32_error());
    }
    Ok(false)
}

/// `encode_code_page_errors` — one character at a time, so the error handler
/// sees the character that failed.
fn encode_errors(
    code_page: u32,
    w_unicode: PyObjectRef,
    code_points: &[u32],
    errors: &str,
) -> Result<Vec<u8>, crate::PyError> {
    let encoding = code_page_name(code_page);
    // `strict` reports the string rather than the character: the walk that
    // would find it only runs for a handler that can answer with something.
    if errors == "strict" {
        return Err(crate::typedef::unicode_encode_error(
            &encoding,
            w_unicode,
            0,
            0,
            ENCODE_REASON,
        ));
    }
    let flags = encode_code_page_flags(code_page, Some(errors));
    let takes_default = uses_default_char(code_page);
    let mut out: Vec<u8> = Vec::with_capacity(code_points.len());
    let mut pos = 0usize;
    while pos < code_points.len() {
        if encode_one(code_page, flags, takes_default, code_points[pos], &mut out)? {
            pos += 1;
            continue;
        }
        let (replacement, newpos) = crate::type_methods::call_registered_encode_error_handler(
            errors,
            &encoding,
            w_unicode,
            code_points.len(),
            pos,
            pos + 1,
            ENCODE_REASON,
        )?;
        match replacement {
            crate::type_methods::EncodeReplacement::Bytes(bytes) => {
                out.extend_from_slice(&bytes);
            }
            // A str replacement is copied as ASCII rather than re-encoded: a
            // handler that answers with anything else has not produced
            // something this code page can be asked to spell.
            crate::type_methods::EncodeReplacement::Str(replacement_points) => {
                for point in replacement_points {
                    if point > 127 {
                        return Err(crate::typedef::unicode_encode_error(
                            &encoding,
                            w_unicode,
                            pos,
                            pos + 1,
                            "unable to encode error handler result to ASCII",
                        ));
                    }
                    out.push(point as u8);
                }
            }
        }
        pos = newpos;
    }
    Ok(out)
}

/// [`encode_code_page`] for a caller holding a name rather than a `str`
/// object, under the one handler that never has to be shown the character it
/// failed on: `replace` answers a `?` per code point, which is written here
/// instead of being fetched from the error registry.
pub fn encode_code_page_replace(code_page: u32, text: &Wtf8) -> Result<Vec<u8>, crate::PyError> {
    let wide: Vec<u16> = text.encode_wide().collect();
    if wide.is_empty() {
        return Ok(Vec::new());
    }
    match encode_strict(code_page, &wide) {
        Ok(bytes) => Ok(bytes),
        Err(CodePageFail::NoTranslation) => {
            let flags = encode_code_page_flags(code_page, Some("replace"));
            let takes_default = uses_default_char(code_page);
            let mut out: Vec<u8> = Vec::with_capacity(wide.len());
            for point in text.code_points() {
                if !encode_one(code_page, flags, takes_default, point.to_u32(), &mut out)? {
                    out.push(b'?');
                }
            }
            Ok(out)
        }
        Err(CodePageFail::Os(error)) => Err(error),
    }
}

/// `PyUnicode_EncodeCodePage`.
pub fn encode_code_page(
    code_page: u32,
    w_unicode: PyObjectRef,
    errors: &str,
) -> Result<Vec<u8>, crate::PyError> {
    let wtf8 = unsafe { pyre_object::w_str_get_wtf8(w_unicode) };
    if wtf8.is_empty() {
        return Ok(Vec::new());
    }
    let wide: Vec<u16> = wtf8.encode_wide().collect();
    match encode_strict(code_page, &wide) {
        Ok(bytes) => Ok(bytes),
        Err(CodePageFail::NoTranslation) => {
            let code_points: Vec<u32> = wtf8.code_points().map(|c| c.to_u32()).collect();
            encode_errors(code_page, w_unicode, &code_points, errors)
        }
        Err(CodePageFail::Os(error)) => Err(error),
    }
}
