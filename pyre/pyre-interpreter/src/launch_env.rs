//! The launcher option block and the environment half of its resolution
//! (`config_read_env_vars`).
//!
//! `app_main.py` reads the `PYTHON*` variables itself, so the folding rules
//! belong beside the `SYS_*` statics they feed rather than in whichever
//! binary happens to own a command line. Two callers share them: the native
//! launcher, which parses an argv and then folds the process environment over
//! it; and the wasm32 embedding, which has neither — `std::env` there is
//! permanently empty, so its host installs the relevant entries through
//! [`set_launch_env`] and folds them over an all-default option block.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// The launcher options, as parsed from a command line and then folded with
/// the environment by [`finalize`]. `app_main.py` keeps the same set in its
/// `options` dict before `sys` initialization reads them back.
#[derive(Clone, Default)]
pub struct LaunchFlags {
    pub inspect: bool,
    pub quiet: bool,
    pub no_site: bool,
    pub no_user_site: bool,
    pub ignore_environment: bool,
    pub isolated: bool,
    pub dev_mode: bool,
    pub warn_default_encoding: bool,
    /// `None` until [`finalize`] resolves it; a command line that named
    /// `-X utf8` carries that value through instead.
    pub utf8_mode: Option<i64>,
    pub safe_path: bool,
    /// `-O` count on the command line; PYTHONOPTIMIZE folds in during finalize.
    pub optimize: i64,
    pub bytes_warning: i64,
    pub dont_write_bytecode: bool,
    pub unbuffered: bool,
    /// Both option lists stay in the host's own spelling until sys module
    /// initialization decodes them, because neither is required to be text the
    /// host can spell in UTF-8: `-W $'ignore\xff'` reaches `sys.warnoptions` as
    /// `'ignore\udcff'`, and PYTHONWARNINGS carries the same bytes.
    pub warnoptions: Vec<std::ffi::OsString>,
    /// `app_main.py` passes the raw PYTHONIOENCODING value to initstdio after
    /// applying -E/-I. Keep it raw until stdio parses the optional errors part.
    pub stdio_encoding: Option<String>,
    /// Every raw `-X` value stays in a list until sys module initialization
    /// turns it into `sys._xoptions`.
    pub xoptions: Vec<std::ffi::OsString>,
}

/// Environment the launcher options resolve against, when the process
/// environment is not it. Installed once at startup; absent means read the
/// process environment through the host seam.
///
/// Values are raw bytes, not `String`: `_Py_GetEnv` tests presence on the
/// undecoded bytes, so a value that is not valid Unicode is still *set*. A
/// table of `String` could not represent that, and the name whose fold depends
/// on it — `PYTHONSAFEPATH` — would silently read as unset.
static LAUNCH_ENV: LazyLock<Mutex<Option<HashMap<String, Vec<u8>>>>> =
    LazyLock::new(|| Mutex::new(None));

/// Supply the launcher-relevant environment explicitly, for an embedding whose
/// own process environment is not the one the program should observe.
///
/// wasm32 has no environment at all — every lookup inside the module returns
/// unset regardless of the host process — so a guest resolving `-P`, `-O` or
/// PYTHONWARNINGS from it reads them all as absent. Passing the entries in
/// restores the fold. Once installed the table is authoritative: a name missing
/// from it is unset, never a fallback to the process environment.
pub fn set_launch_env(entries: impl IntoIterator<Item = (String, Vec<u8>)>) {
    *LAUNCH_ENV.lock().unwrap() = Some(entries.into_iter().collect());
}

/// The names [`finalize`] reads, for a host that has to forward them to an
/// embedding with no environment of its own. Locale variables are included
/// because an unset chain resolves to the C locale, which coerces UTF-8 mode.
pub const LAUNCH_ENV_NAMES: &[&str] = &[
    "PYTHONSAFEPATH",
    "PYTHONNOUSERSITE",
    "PYTHONUNBUFFERED",
    "PYTHONDEVMODE",
    "PYTHONINSPECT",
    "PYTHONOPTIMIZE",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONUTF8",
    "PYTHONWARNDEFAULTENCODING",
    "PYTHONWARNINGS",
    "PYTHONIOENCODING",
    "LC_ALL",
    "LC_CTYPE",
    "LANG",
];

/// The raw bytes of a variable, from the installed table when there is one and
/// otherwise from the process environment through the host seam. A seam error
/// reads as unset, the same as a name the environment does not carry.
fn read_raw(name: &str) -> Option<Vec<u8>> {
    if let Some(table) = LAUNCH_ENV.lock().unwrap().as_ref() {
        return table.get(name).cloned();
    }
    crate::host_seam::getenv(name.as_bytes()).ok().flatten()
}

/// The decoded value, or `None` when unset *or* not valid Unicode — the
/// `env::var` contract. Only the free-text names resolve under it
/// (PYTHONWARNINGS, PYTHONIOENCODING and the locale chain); every name whose
/// fold turns on presence or on an exact spelling reads bytes instead, so that
/// an undecodable value stays distinguishable from an absent one.
fn read(name: &str) -> Option<String> {
    read_raw(name).and_then(|value| String::from_utf8(value).ok())
}

/// Environment bytes in the host's own spelling. The seam hands back what the
/// platform stores: bytes on unix, where any byte is legal, and the WTF-8 form
/// of the wide value on Windows, whose unpaired surrogates re-encode to the
/// code units they came from.
fn os_string_from_bytes(value: &[u8]) -> std::ffi::OsString {
    crate::gateway::os_string_from_fs_bytes(value)
}

/// Presence of a variable, without decoding it. `_Py_GetEnv` tests the raw
/// bytes, so a value that is not valid Unicode still counts as set. Both paths
/// preserve that: the seam hands back bytes, and the installed table stores
/// bytes for this reason.
fn is_set_nonempty(name: &str) -> bool {
    read_raw(name).is_some_and(|value| !value.is_empty())
}

/// Whether the effective `LC_CTYPE` is the legacy C/POSIX locale, which is what
/// coerces utf8_mode to 1; every named locale (en_US, C.UTF-8, …) leaves it 0.
fn locale_implies_utf8_mode() -> bool {
    // `_Py_SetLocaleFromEnv(LC_CTYPE)` and read the answer back, so the value
    // tested is the locale the C library actually installed rather than the
    // variables that asked for it. The two differ whenever the environment
    // names a locale the system cannot set: `setlocale` keeps C there, and
    // utf8_mode has to follow the installed locale, not the unusable name.
    //
    // Only when the process environment is the authority — an embedding that
    // supplied its own table is describing an environment this process does not
    // have, and `setlocale("")` would answer from the wrong one, so that path
    // keeps the string cascade below. wasm32 has no locale database at all and
    // reaches the cascade the same way.
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    if LAUNCH_ENV.lock().unwrap().is_none() {
        rustpython_host_env::locale::setlocale(libc::LC_CTYPE, Some(c""));
        let effective = rustpython_host_env::locale::setlocale(libc::LC_CTYPE, None);
        return matches!(effective.as_deref(), None | Some(b"C") | Some(b"POSIX"));
    }
    // An empty variable is treated as unset (`setlocale` POSIX semantics) and
    // falls through to the next, so `LC_ALL= LC_CTYPE=en_US.UTF-8` resolves to
    // en_US.UTF-8, not C.
    let read = |name: &str| read(name).filter(|v| !v.is_empty());
    let locale = read("LC_ALL")
        .or_else(|| read("LC_CTYPE"))
        .or_else(|| read("LANG"));
    // Only the legacy C/POSIX locale — or a fully unset chain, which resolves to
    // C — coerces utf8_mode to 1; every named locale (en_US, C.UTF-8, …) leaves
    // it 0.
    matches!(locale.as_deref(), None | Some("C") | Some("POSIX"))
}

/// Detail of a `preconfig_init_utf8_mode` failure, for the caller to report the
/// way its embedding reports a fatal pre-init error.
pub struct PreConfigError(pub &'static str);

fn resolve_utf8_mode(flags: &LaunchFlags) -> Result<i64, PreConfigError> {
    if let Some(value) = flags.utf8_mode {
        return Ok(value);
    }
    if !flags.ignore_environment {
        // Compared as bytes: a value that does not decode is *invalid*, which is
        // fatal, not absent, which would silently fall through to the locale.
        if let Some(value) = read_raw("PYTHONUTF8")
            && !value.is_empty()
        {
            return match value.as_slice() {
                b"0" => Ok(0),
                b"1" => Ok(1),
                _ => Err(PreConfigError(
                    "invalid PYTHONUTF8 environment variable value",
                )),
            };
        }
    }
    Ok(if locale_implies_utf8_mode() { 1 } else { 0 })
}

/// `_Py_get_env_flag`: an integer environment flag. An unset or empty value is
/// absent; a clean non-negative integer is used as-is; anything else (trailing
/// junk, negative, overflow) counts as 1.
fn env_int_flag(name: &str) -> Option<u32> {
    let raw = read_raw(name)?;
    if raw.is_empty() {
        return None;
    }
    // Read the bytes, not a decoded string: `_Py_str_to_int` rejects undecodable
    // input exactly the way it rejects trailing junk, so both land on 1 rather
    // than on "unset".
    let parsed = std::str::from_utf8(&raw)
        .ok()
        .and_then(|value| value.parse::<i64>().ok());
    let value = match parsed {
        Some(v) if (0..=i64::from(u32::MAX)).contains(&v) => v as u32,
        _ => 1,
    };
    Some(value)
}

/// `-O` count folded with PYTHONOPTIMIZE (`config_init_optimization_level`):
/// the effective level is the larger of the two.  The level is kept as a wide
/// integer so `sys.flags.optimize` mirrors a large `PYTHONOPTIMIZE` verbatim;
/// the compiler clamps it into a byte at read time.
fn resolve_optimize(flags: &LaunchFlags) -> i64 {
    let mut level = flags.optimize;
    if !flags.ignore_environment
        && let Some(v) = env_int_flag("PYTHONOPTIMIZE")
    {
        level = level.max(i64::from(v));
    }
    level
}

/// A boolean option folded with an *integer* environment flag: the variable
/// enables it for any value `_Py_get_env_flag` resolves nonzero, so an explicit
/// `NAME=0` leaves it off. The fold only ever sets, so the command-line spelling
/// survives both `-E` and a zero-valued variable.
fn fold_int_flag(flags: &LaunchFlags, set: bool, name: &str) -> bool {
    set || (!flags.ignore_environment && env_int_flag(name).is_some_and(|v| v != 0))
}

/// A boolean option folded with a *presence* environment flag, read through
/// `_Py_GetEnv` rather than `_Py_get_env_flag`: any non-empty value enables it —
/// `"0"` included — while an empty value counts as unset.
fn fold_presence_flag(flags: &LaunchFlags, set: bool, name: &str) -> bool {
    set || (!flags.ignore_environment && is_set_nonempty(name))
}

/// Fold the environment over the options a command line named. An embedding
/// without a command line passes `LaunchFlags::default()`.
pub fn finalize(mut flags: LaunchFlags) -> Result<LaunchFlags, PreConfigError> {
    flags.utf8_mode = Some(resolve_utf8_mode(&flags)?);
    flags.optimize = resolve_optimize(&flags);
    // Which of the two shapes a name takes is per-variable, not a category:
    // PYTHONSAFEPATH and PYTHONINSPECT enable on a bare `=0`, PYTHONNOUSERSITE
    // and PYTHONUNBUFFERED do not.
    flags.dont_write_bytecode =
        fold_int_flag(&flags, flags.dont_write_bytecode, "PYTHONDONTWRITEBYTECODE");
    flags.no_user_site = fold_int_flag(&flags, flags.no_user_site, "PYTHONNOUSERSITE");
    flags.unbuffered = fold_int_flag(&flags, flags.unbuffered, "PYTHONUNBUFFERED");
    flags.safe_path = fold_presence_flag(&flags, flags.safe_path, "PYTHONSAFEPATH");
    flags.dev_mode = fold_presence_flag(&flags, flags.dev_mode, "PYTHONDEVMODE");
    flags.inspect = fold_presence_flag(&flags, flags.inspect, "PYTHONINSPECT");
    // app_main.py:773-774 — the variable enables it on any non-empty value,
    // `0` included, the same way `-X warn_default_encoding` does.
    flags.warn_default_encoding = fold_presence_flag(
        &flags,
        flags.warn_default_encoding,
        "PYTHONWARNDEFAULTENCODING",
    );
    flags.stdio_encoding = if flags.ignore_environment {
        None
    } else {
        read("PYTHONIOENCODING")
    };
    // pypy/interpreter/app_main.py:892-906 — lowest-precedence entries first;
    // the warnings module installs later entries ahead of earlier ones.
    let mut warnoptions: Vec<std::ffi::OsString> = Vec::new();
    if flags.dev_mode {
        warnoptions.push("default".into());
    }
    if !flags.ignore_environment {
        // Read as bytes: a filter is free text, so `read`'s `env::var`
        // contract would drop the whole variable for one undecodable byte
        // rather than carry the entry `-W` would have carried.
        if let Some(value) = read_raw("PYTHONWARNINGS")
            && !value.is_empty()
        {
            warnoptions.extend(value.split(|&b| b == b',').map(os_string_from_bytes));
        }
    }
    warnoptions.append(&mut flags.warnoptions);
    if flags.bytes_warning > 0 {
        warnoptions.push(if flags.bytes_warning > 1 {
            "error::BytesWarning".into()
        } else {
            "default::BytesWarning".into()
        });
    }
    flags.warnoptions = warnoptions;
    Ok(flags)
}
