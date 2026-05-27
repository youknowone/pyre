//! _locale implementation — PyPy: pypy/module/_locale/interp_locale.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::DictStorage;

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

/// `_locale` C-extension stub — PyPy: pypy/module/_locale/.
///
/// Provides the 'C' locale defaults so locale.py's `from _locale import *`
/// succeeds and Lib/locale.py exposes working `localeconv`/`setlocale`.
/// This mirrors the `except ImportError` fallback in the stdlib's
/// `locale` module, but routed through pyre's builtin-module registry
/// so a single import succeeds.
pub fn register_module(ns: &mut DictStorage) {
    // Locale category constants sourced from libc so the values match
    // the host (Linux: LC_CTYPE=0; macOS: LC_ALL=0, LC_CTYPE=2; ...).
    #[cfg(unix)]
    {
        crate::dict_storage_store(
            ns,
            "LC_CTYPE",
            pyre_object::w_int_new(libc::LC_CTYPE as i64),
        );
        crate::dict_storage_store(
            ns,
            "LC_NUMERIC",
            pyre_object::w_int_new(libc::LC_NUMERIC as i64),
        );
        crate::dict_storage_store(ns, "LC_TIME", pyre_object::w_int_new(libc::LC_TIME as i64));
        crate::dict_storage_store(
            ns,
            "LC_COLLATE",
            pyre_object::w_int_new(libc::LC_COLLATE as i64),
        );
        crate::dict_storage_store(
            ns,
            "LC_MONETARY",
            pyre_object::w_int_new(libc::LC_MONETARY as i64),
        );
        crate::dict_storage_store(
            ns,
            "LC_MESSAGES",
            pyre_object::w_int_new(libc::LC_MESSAGES as i64),
        );
        crate::dict_storage_store(ns, "LC_ALL", pyre_object::w_int_new(libc::LC_ALL as i64));
    }
    #[cfg(not(unix))]
    {
        crate::dict_storage_store(ns, "LC_CTYPE", pyre_object::w_int_new(0));
        crate::dict_storage_store(ns, "LC_NUMERIC", pyre_object::w_int_new(1));
        crate::dict_storage_store(ns, "LC_TIME", pyre_object::w_int_new(2));
        crate::dict_storage_store(ns, "LC_COLLATE", pyre_object::w_int_new(3));
        crate::dict_storage_store(ns, "LC_MONETARY", pyre_object::w_int_new(4));
        crate::dict_storage_store(ns, "LC_MESSAGES", pyre_object::w_int_new(5));
        crate::dict_storage_store(ns, "LC_ALL", pyre_object::w_int_new(6));
    }
    crate::dict_storage_store(ns, "CHAR_MAX", pyre_object::w_int_new(127));
    #[cfg(all(
        unix,
        not(any(target_os = "ios", target_os = "android", target_os = "redox"))
    ))]
    {
        crate::dict_storage_store(ns, "CODESET", pyre_object::w_int_new(libc::CODESET as i64));
    }
    // `interp_locale.py:11 W_Error = _new_exception('Error', W_Exception, 'locale error')`
    let exception_base = crate::builtins::lookup_exc_class("Exception")
        .expect("Exception must be installed before _locale init");
    let w_error = crate::builtins::make_exc_type(
        "_locale.Error",
        crate::builtins::exc_exception_new,
        exception_base,
    );
    crate::dict_storage_store(ns, "Error", w_error);

    // localeconv() — returns the 'C' locale parameters as a dict.
    crate::dict_storage_store(
        ns,
        "localeconv",
        crate::make_builtin_function_with_arity(
            "localeconv",
            |_| {
                let d = pyre_object::w_dict_new();
                unsafe {
                    pyre_object::w_dict_setitem_str(
                        d,
                        "grouping",
                        pyre_object::w_list_new(vec![pyre_object::w_int_new(127)]),
                    );
                    pyre_object::w_dict_setitem_str(
                        d,
                        "currency_symbol",
                        pyre_object::w_str_new(""),
                    );
                    pyre_object::w_dict_setitem_str(d, "n_sign_posn", pyre_object::w_int_new(127));
                    pyre_object::w_dict_setitem_str(
                        d,
                        "p_cs_precedes",
                        pyre_object::w_int_new(127),
                    );
                    pyre_object::w_dict_setitem_str(
                        d,
                        "n_cs_precedes",
                        pyre_object::w_int_new(127),
                    );
                    pyre_object::w_dict_setitem_str(
                        d,
                        "mon_grouping",
                        pyre_object::w_list_new(vec![]),
                    );
                    pyre_object::w_dict_setitem_str(
                        d,
                        "n_sep_by_space",
                        pyre_object::w_int_new(127),
                    );
                    pyre_object::w_dict_setitem_str(
                        d,
                        "decimal_point",
                        pyre_object::w_str_new("."),
                    );
                    pyre_object::w_dict_setitem_str(d, "negative_sign", pyre_object::w_str_new(""));
                    pyre_object::w_dict_setitem_str(d, "positive_sign", pyre_object::w_str_new(""));
                    pyre_object::w_dict_setitem_str(
                        d,
                        "p_sep_by_space",
                        pyre_object::w_int_new(127),
                    );
                    pyre_object::w_dict_setitem_str(
                        d,
                        "int_curr_symbol",
                        pyre_object::w_str_new(""),
                    );
                    pyre_object::w_dict_setitem_str(d, "p_sign_posn", pyre_object::w_int_new(127));
                    pyre_object::w_dict_setitem_str(d, "thousands_sep", pyre_object::w_str_new(""));
                    pyre_object::w_dict_setitem_str(
                        d,
                        "mon_thousands_sep",
                        pyre_object::w_str_new(""),
                    );
                    pyre_object::w_dict_setitem_str(d, "frac_digits", pyre_object::w_int_new(127));
                    pyre_object::w_dict_setitem_str(
                        d,
                        "mon_decimal_point",
                        pyre_object::w_str_new(""),
                    );
                    pyre_object::w_dict_setitem_str(
                        d,
                        "int_frac_digits",
                        pyre_object::w_int_new(127),
                    );
                }
                Ok(d)
            },
            0,
        ),
    );
    crate::dict_storage_store(
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
                            .map_err(|_| crate::PyError::value_error("embedded null"))?,
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
    crate::dict_storage_store(
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
                            return Ok(pyre_object::w_str_new(&String::from_utf8_lossy(&bytes)));
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
                    let s = unsafe { std::ffi::CStr::from_ptr(p) };
                    return Ok(pyre_object::w_str_new(&s.to_string_lossy()));
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
    crate::dict_storage_store(
        ns,
        "strcoll",
        crate::make_builtin_function_with_arity(
            "strcoll",
            |args| {
                #[cfg(all(unix, feature = "host_env"))]
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
                        .map_err(|_| crate::PyError::value_error("embedded null"))?;
                    let c2 = std::ffi::CString::new(s2.as_bytes())
                        .map_err(|_| crate::PyError::value_error("embedded null"))?;
                    return Ok(pyre_object::w_int_new(
                        rustpython_host_env::locale::strcoll(&c1, &c2) as i64,
                    ));
                }
                #[cfg(not(all(unix, feature = "host_env")))]
                {
                    if args.len() < 2
                        || !unsafe { pyre_object::is_str(args[0]) && pyre_object::is_str(args[1]) }
                    {
                        return Err(crate::PyError::type_error(
                            "strcoll: arguments must be strings",
                        ));
                    }
                    // No libc collation available — fall back to
                    // lexical bytewise comparison.  Pure computation,
                    // no I/O, so the sandbox principle is unaffected.
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
    crate::dict_storage_store(
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
                Ok(s)
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "getencoding",
        crate::make_builtin_function_with_arity(
            "getencoding",
            |_| Ok(pyre_object::w_str_new("utf-8")),
            0,
        ),
    );
}
