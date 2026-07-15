//! _locale implementation — PyPy: pypy/module/_locale/interp_locale.py
//!
//! Verbatim move of the inline block previously in importing.rs.

// Under sandbox, name libc through the seam facade so any direct syscall call
// in this module is a compile error (only types/constants/pure fns resolve).
#[cfg(feature = "sandbox")]
use crate::host_seam::sys as libc;

/// Raise `_locale.Error` with the supplied message.  Mirrors
/// `interp_locale.py:15-20 make_error`.
fn locale_error(message: &str) -> crate::PyError {
    let cls = crate::builtins::lookup_exc_class("_locale.Error")
        .or_else(|| crate::builtins::lookup_exc_class("Exception"))
        .expect("Exception must be installed");
    let args = vec![cls, pyre_object::w_str_new(message)];
    let exc = crate::builtins::exc_exception_new(&args)
        .expect("exc_exception_new is infallible for str args");
    let mut err = crate::PyError::value_error(message);
    err.exc_object = exc;
    err
}

/// Numeric/monetary locale parameters decoded into owned buffers, the
/// `lconv` fields `localeconv` exposes (`interp_locale.py:48`).
struct LocaleConvData {
    decimal_point: Vec<u8>,
    thousands_sep: Vec<u8>,
    grouping: Vec<i64>,
    int_curr_symbol: Vec<u8>,
    currency_symbol: Vec<u8>,
    mon_decimal_point: Vec<u8>,
    mon_thousands_sep: Vec<u8>,
    mon_grouping: Vec<i64>,
    positive_sign: Vec<u8>,
    negative_sign: Vec<u8>,
    int_frac_digits: i64,
    frac_digits: i64,
    p_cs_precedes: i64,
    p_sep_by_space: i64,
    n_cs_precedes: i64,
    n_sep_by_space: i64,
    p_sign_posn: i64,
    n_sign_posn: i64,
}

/// Build the `localeconv()` result dict.  String fields decode the host
/// bytes via `charp2uni` (utf-8 + surrogateescape, `interp_locale.py:42`);
/// grouping lists already carry the trailing 0 that `_w_copy_grouping`
/// appends to a non-empty grouping (`interp_locale.py:36-40`).
fn localeconv_to_dict(c: &LocaleConvData) -> pyre_object::PyObjectRef {
    let d = pyre_object::w_dict_new();
    let put_str = |k: &str, b: &[u8]| unsafe {
        pyre_object::w_dict_setitem_str(d, k, crate::typedef::charp2uni(b));
    };
    let put_int = |k: &str, v: i64| unsafe {
        pyre_object::w_dict_setitem_str(d, k, pyre_object::w_int_new(v));
    };
    let put_grouping = |k: &str, v: &[i64]| unsafe {
        pyre_object::w_dict_setitem_str(
            d,
            k,
            pyre_object::w_list_new(v.iter().map(|&x| pyre_object::w_int_new(x)).collect()),
        );
    };
    put_str("decimal_point", &c.decimal_point);
    put_str("thousands_sep", &c.thousands_sep);
    put_grouping("grouping", &c.grouping);
    put_str("int_curr_symbol", &c.int_curr_symbol);
    put_str("currency_symbol", &c.currency_symbol);
    put_str("mon_decimal_point", &c.mon_decimal_point);
    put_str("mon_thousands_sep", &c.mon_thousands_sep);
    put_grouping("mon_grouping", &c.mon_grouping);
    put_str("positive_sign", &c.positive_sign);
    put_str("negative_sign", &c.negative_sign);
    put_int("int_frac_digits", c.int_frac_digits);
    put_int("frac_digits", c.frac_digits);
    put_int("p_cs_precedes", c.p_cs_precedes);
    put_int("p_sep_by_space", c.p_sep_by_space);
    put_int("n_cs_precedes", c.n_cs_precedes);
    put_int("n_sep_by_space", c.n_sep_by_space);
    put_int("p_sign_posn", c.p_sign_posn);
    put_int("n_sign_posn", c.n_sign_posn);
    d
}

/// POSIX "C" locale `localeconv` defaults for builds without libc; the
/// char-typed fields default to `CHAR_MAX` (no value provided).
#[cfg(not(all(unix, feature = "host_env")))]
fn c_locale_conv() -> LocaleConvData {
    const CHAR_MAX: i64 = 127;
    LocaleConvData {
        decimal_point: b".".to_vec(),
        thousands_sep: Vec::new(),
        grouping: Vec::new(),
        int_curr_symbol: Vec::new(),
        currency_symbol: Vec::new(),
        mon_decimal_point: Vec::new(),
        mon_thousands_sep: Vec::new(),
        mon_grouping: Vec::new(),
        positive_sign: Vec::new(),
        negative_sign: Vec::new(),
        int_frac_digits: CHAR_MAX,
        frac_digits: CHAR_MAX,
        p_cs_precedes: CHAR_MAX,
        p_sep_by_space: CHAR_MAX,
        n_cs_precedes: CHAR_MAX,
        n_sep_by_space: CHAR_MAX,
        p_sign_posn: CHAR_MAX,
        n_sign_posn: CHAR_MAX,
    }
}

/// `_locale` C-extension stub — PyPy: pypy/module/_locale/.
///
/// Provides the 'C' locale defaults so locale.py's `from _locale import *`
/// succeeds and Lib/locale.py exposes working `localeconv`/`setlocale`.
/// This mirrors the `except ImportError` fallback in the stdlib's
/// `locale` module, but routed through pyre's builtin-module registry
/// so a single import succeeds.
pub fn register_module(ns: pyre_object::PyObjectRef) {
    // Locale category constants sourced from libc so the values match
    // the host (Linux: LC_CTYPE=0; macOS: LC_ALL=0, LC_CTYPE=2; ...).
    #[cfg(unix)]
    {
        crate::module_ns_store(
            ns,
            "LC_CTYPE",
            pyre_object::w_int_new(libc::LC_CTYPE as i64),
        );
        crate::module_ns_store(
            ns,
            "LC_NUMERIC",
            pyre_object::w_int_new(libc::LC_NUMERIC as i64),
        );
        crate::module_ns_store(ns, "LC_TIME", pyre_object::w_int_new(libc::LC_TIME as i64));
        crate::module_ns_store(
            ns,
            "LC_COLLATE",
            pyre_object::w_int_new(libc::LC_COLLATE as i64),
        );
        crate::module_ns_store(
            ns,
            "LC_MONETARY",
            pyre_object::w_int_new(libc::LC_MONETARY as i64),
        );
        crate::module_ns_store(
            ns,
            "LC_MESSAGES",
            pyre_object::w_int_new(libc::LC_MESSAGES as i64),
        );
        crate::module_ns_store(ns, "LC_ALL", pyre_object::w_int_new(libc::LC_ALL as i64));
    }
    #[cfg(not(unix))]
    {
        crate::module_ns_store(ns, "LC_CTYPE", pyre_object::w_int_new(0));
        crate::module_ns_store(ns, "LC_NUMERIC", pyre_object::w_int_new(1));
        crate::module_ns_store(ns, "LC_TIME", pyre_object::w_int_new(2));
        crate::module_ns_store(ns, "LC_COLLATE", pyre_object::w_int_new(3));
        crate::module_ns_store(ns, "LC_MONETARY", pyre_object::w_int_new(4));
        crate::module_ns_store(ns, "LC_MESSAGES", pyre_object::w_int_new(5));
        crate::module_ns_store(ns, "LC_ALL", pyre_object::w_int_new(6));
    }
    crate::module_ns_store(ns, "CHAR_MAX", pyre_object::w_int_new(127));
    #[cfg(all(
        unix,
        not(any(target_os = "ios", target_os = "android", target_os = "redox"))
    ))]
    {
        crate::module_ns_store(ns, "CODESET", pyre_object::w_int_new(libc::CODESET as i64));
    }
    // `interp_locale.py:11 W_Error = _new_exception('Error', W_Exception, 'locale error')`
    let exception_base = crate::builtins::lookup_exc_class("Exception")
        .expect("Exception must be installed before _locale init");
    let w_error = crate::builtins::make_exc_type(
        "_locale.Error",
        crate::builtins::exc_exception_new,
        exception_base,
    );
    crate::module_ns_store(ns, "Error", w_error);

    // localeconv() — numeric/monetary parameters of the current locale.
    // Reads the host locale DB; under sandbox the stub override below replaces
    // it, so the real body (and its libc/host_env calls) is compiled out.
    #[cfg(not(feature = "sandbox"))]
    crate::module_ns_store(
        ns,
        "localeconv",
        crate::make_builtin_function_with_arity(
            "localeconv",
            |_| {
                #[cfg(all(unix, feature = "host_env"))]
                {
                    let lc = rustpython_host_env::locale::localeconv_data();
                    // `_w_copy_grouping` (`interp_locale.py:36-40`): every byte
                    // of the C grouping string up to its NUL is one group size
                    // (a `CHAR_MAX` element stays `127`), then a trailing `0`
                    // is appended to a non-empty list. Read the grouping
                    // straight from `localeconv()` because the host helper
                    // stops at the first `CHAR_MAX`, dropping it.
                    let grouping_of = |ptr: *const libc::c_char| -> Vec<i64> {
                        let mut v: Vec<i64> = Vec::new();
                        if !ptr.is_null() {
                            let mut cur = ptr;
                            unsafe {
                                while *cur != 0 {
                                    v.push(*cur as u8 as i64);
                                    cur = cur.add(1);
                                }
                            }
                        }
                        if !v.is_empty() {
                            v.push(0);
                        }
                        v
                    };
                    let raw = unsafe { libc::localeconv() };
                    let (grouping, mon_grouping) = if raw.is_null() {
                        (Vec::new(), Vec::new())
                    } else {
                        unsafe {
                            (
                                grouping_of((*raw).grouping),
                                grouping_of((*raw).mon_grouping),
                            )
                        }
                    };
                    let data = LocaleConvData {
                        decimal_point: lc.decimal_point.clone(),
                        thousands_sep: lc.thousands_sep.clone(),
                        grouping,
                        int_curr_symbol: lc.int_curr_symbol.clone(),
                        currency_symbol: lc.currency_symbol.clone(),
                        mon_decimal_point: lc.mon_decimal_point.clone(),
                        mon_thousands_sep: lc.mon_thousands_sep.clone(),
                        mon_grouping,
                        positive_sign: lc.positive_sign.clone(),
                        negative_sign: lc.negative_sign.clone(),
                        int_frac_digits: lc.int_frac_digits as i64,
                        frac_digits: lc.frac_digits as i64,
                        p_cs_precedes: lc.p_cs_precedes as i64,
                        p_sep_by_space: lc.p_sep_by_space as i64,
                        n_cs_precedes: lc.n_cs_precedes as i64,
                        n_sep_by_space: lc.n_sep_by_space as i64,
                        p_sign_posn: lc.p_sign_posn as i64,
                        n_sign_posn: lc.n_sign_posn as i64,
                    };
                    Ok(localeconv_to_dict(&data))
                }
                #[cfg(not(all(unix, feature = "host_env")))]
                {
                    Ok(localeconv_to_dict(&c_locale_conv()))
                }
            },
            0,
        ),
    );
    // setlocale() mutates/reads the host locale (and $LANG/$LC_*); stubbed under
    // sandbox, so the real body is compiled out.
    #[cfg(not(feature = "sandbox"))]
    crate::module_ns_store(
        ns,
        "setlocale",
        crate::make_builtin_function("setlocale", |args| {
            // Argument contract is shared between host_env and the
            // no-libc fallback so invalid calls always raise instead of
            // silently echoing "C".
            if args.is_empty() || args.len() > 2 {
                return Err(crate::PyError::type_error(
                    "setlocale() takes 1 or 2 arguments",
                ));
            }
            if !unsafe { pyre_object::is_int(args[0]) } {
                return Err(crate::PyError::type_error(
                    "setlocale: category must be an integer",
                ));
            }
            let locale_str: Option<String> =
                if args.len() >= 2 && !unsafe { pyre_object::is_none(args[1]) } {
                    if !unsafe { pyre_object::is_str(args[1]) } {
                        return Err(crate::PyError::type_error(
                            "setlocale: locale must be a string or None",
                        ));
                    }
                    Some(unsafe { pyre_object::w_str_get_value(args[1]).to_string() })
                } else {
                    None
                };
            #[cfg(all(unix, feature = "host_env"))]
            {
                let cat = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                let c_locale = match locale_str.as_ref() {
                    Some(s) => Some(
                        std::ffi::CString::new(s.as_bytes())
                            .map_err(|_| crate::PyError::value_error("embedded null character"))?,
                    ),
                    None => None,
                };
                let out = rustpython_host_env::locale::setlocale(cat, c_locale.as_deref());
                match out {
                    Some(bytes) => Ok(pyre_object::w_str_new(&String::from_utf8_lossy(&bytes))),
                    None => Err(locale_error("unsupported locale setting")),
                }
            }
            #[cfg(not(all(unix, feature = "host_env")))]
            {
                // No libc available — every valid call resolves to the
                // POSIX "C" locale.  `locale_str` is dropped on purpose.
                let _ = locale_str;
                Ok(pyre_object::w_str_new("C"))
            }
        }),
    );
    // nl_langinfo() reads the active-locale codeset/DB; stubbed under sandbox,
    // so the real body is compiled out.
    #[cfg(not(feature = "sandbox"))]
    crate::module_ns_store(
        ns,
        "nl_langinfo",
        crate::make_builtin_function_with_arity(
            "nl_langinfo",
            |args| {
                #[cfg(all(
                    unix,
                    feature = "host_env",
                    not(any(target_os = "ios", target_os = "android", target_os = "redox"))
                ))]
                {
                    let item = if args.is_empty() {
                        libc::CODESET
                    } else {
                        if !unsafe { pyre_object::is_int(args[0]) } {
                            return Err(crate::PyError::type_error(
                                "nl_langinfo: item must be an integer",
                            ));
                        }
                        (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::nl_item
                    };
                    if item == libc::CODESET {
                        if let Some(bytes) = rustpython_host_env::locale::nl_langinfo_codeset() {
                            return Ok(crate::typedef::charp2uni(&bytes));
                        }
                    }
                    // `interp_locale.py:151-154` — unknown items raise
                    // ValueError("unsupported langinfo constant").  POSIX
                    // nl_langinfo never returns NULL for valid items, so a
                    // null return is treated as the unsupported case.
                    let p = unsafe { libc::nl_langinfo(item) };
                    if p.is_null() {
                        return Err(crate::PyError::value_error("unsupported langinfo constant"));
                    }
                    // `interp_locale.py:153` decodes via utf-8 + surrogateescape.
                    let s = unsafe { std::ffi::CStr::from_ptr(p) };
                    return Ok(crate::typedef::charp2uni(s.to_bytes()));
                }
                #[cfg(not(all(
                    unix,
                    feature = "host_env",
                    not(any(target_os = "ios", target_os = "android", target_os = "redox"))
                )))]
                {
                    // No langinfo available → every constant counts as
                    // unsupported, matching `interp_locale.py:151-154`.
                    let _ = args;
                    Err(crate::PyError::value_error("unsupported langinfo constant"))
                }
            },
            1,
        ),
    );
    crate::module_ns_store(
        ns,
        "strcoll",
        crate::make_builtin_function_with_arity(
            "strcoll",
            |args| {
                #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
                {
                    if args.len() < 2
                        || !unsafe { pyre_object::is_str(args[0]) && pyre_object::is_str(args[1]) }
                    {
                        return Err(crate::PyError::type_error(
                            "strcoll: arguments must be strings",
                        ));
                    }
                    let s1 = unsafe { pyre_object::w_str_get_value(args[0]).to_string() };
                    let s2 = unsafe { pyre_object::w_str_get_value(args[1]).to_string() };
                    let c1 = std::ffi::CString::new(s1.as_bytes())
                        .map_err(|_| crate::PyError::value_error("embedded null character"))?;
                    let c2 = std::ffi::CString::new(s2.as_bytes())
                        .map_err(|_| crate::PyError::value_error("embedded null character"))?;
                    return Ok(pyre_object::w_int_new(
                        rustpython_host_env::locale::strcoll(&c1, &c2) as i64,
                    ));
                }
                #[cfg(not(all(unix, feature = "host_env", not(feature = "sandbox"))))]
                {
                    if args.len() < 2
                        || !unsafe { pyre_object::is_str(args[0]) && pyre_object::is_str(args[1]) }
                    {
                        return Err(crate::PyError::type_error(
                            "strcoll: arguments must be strings",
                        ));
                    }
                    // No libc collation available (or sandbox build) — fall back
                    // to lexical bytewise comparison.  Pure computation, no I/O;
                    // under sandbox this keeps the fixed "C" collation and never
                    // calls host libc collation, which would leak host LC_COLLATE.
                    let s1 = unsafe { pyre_object::w_str_get_value(args[0]).to_string() };
                    let s2 = unsafe { pyre_object::w_str_get_value(args[1]).to_string() };
                    let ord = match s1.as_str().cmp(s2.as_str()) {
                        std::cmp::Ordering::Less => -1,
                        std::cmp::Ordering::Equal => 0,
                        std::cmp::Ordering::Greater => 1,
                    };
                    Ok(pyre_object::w_int_new(ord))
                }
            },
            2,
        ),
    );
    crate::module_ns_store(
        ns,
        "strxfrm",
        crate::make_builtin_function_with_arity(
            "strxfrm",
            |args| {
                let s = args[0];
                if !unsafe { pyre_object::is_str(s) } {
                    return Err(crate::PyError::type_error(
                        "strxfrm() argument must be str",
                    ));
                }
                #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
                {
                    let sv = unsafe { pyre_object::w_str_get_value(s).to_string() };
                    let c = std::ffi::CString::new(sv.as_bytes())
                        .map_err(|_| crate::PyError::value_error("embedded null character"))?;
                    let out = rustpython_host_env::locale::strxfrm(&c, sv.len() + 1);
                    // `interp_locale.py:139` returns `space.newtext(val)` —
                    // a plain utf-8 decode (lossy), matching `setlocale`;
                    // unlike `localeconv`/`nl_langinfo` it does not apply
                    // surrogateescape.
                    Ok(pyre_object::w_str_new(&String::from_utf8_lossy(&out)))
                }
                #[cfg(not(all(unix, feature = "host_env", not(feature = "sandbox"))))]
                {
                    // No libc collation available (or sandbox build) — the
                    // transform is identity, keeping the fixed "C" locale and
                    // never reaching host libc strxfrm.
                    Ok(s)
                }
            },
            1,
        ),
    );
    crate::module_ns_store(
        ns,
        "getencoding",
        crate::make_builtin_function_with_arity(
            "getencoding",
            |_| Ok(pyre_object::w_str_new("utf-8")),
            0,
        ),
    );
    // setlocale/localeconv/nl_langinfo read the host locale database (and the
    // active $LANG/$LC_* environment); stub them so the sandbox observes only
    // the fixed "C" locale defaults already exposed above, never host state.
    #[cfg(feature = "sandbox")]
    {
        fn locale_unavailable(
            _: &[pyre_object::PyObjectRef],
        ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
            Err(crate::host_seam::stub("this locale function"))
        }
        for name in ["setlocale", "localeconv", "nl_langinfo"] {
            crate::module_ns_store(
                ns,
                name,
                crate::make_builtin_function(name, locale_unavailable),
            );
        }
    }
}
