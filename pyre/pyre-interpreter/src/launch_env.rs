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
#[derive(Clone)]
pub struct LaunchFlags {
    pub inspect: bool,
    pub quiet: bool,
    pub no_site: bool,
    pub no_user_site: bool,
    pub ignore_environment: bool,
    pub isolated: bool,
    pub dev_mode: bool,
    /// `None` until [`finalize`] resolves it; a command line that named
    /// `-X utf8` carries that value through instead.
    pub utf8_mode: Option<i64>,
    pub safe_path: bool,
    /// `-O` count on the command line; PYTHONOPTIMIZE folds in during finalize.
    pub optimize: i64,
    pub bytes_warning: i64,
    pub dont_write_bytecode: bool,
    pub unbuffered: bool,
    pub warnoptions: Vec<String>,
    /// `app_main.py` passes the raw PYTHONIOENCODING value to initstdio after
    /// applying -E/-I. Keep it raw until stdio parses the optional errors part.
    pub stdio_encoding: Option<String>,
    /// Every raw `-X` value stays in a list until sys module initialization
    /// turns it into `sys._xoptions`.
    pub xoptions: Vec<String>,
}

impl Default for LaunchFlags {
    fn default() -> Self {
        Self {
            inspect: false,
            quiet: false,
            no_site: false,
            no_user_site: false,
            ignore_environment: false,
            isolated: false,
            dev_mode: false,
            utf8_mode: None,
            safe_path: false,
            optimize: 0,
            bytes_warning: 0,
            dont_write_bytecode: false,
            unbuffered: false,
            warnoptions: Vec::new(),
            stdio_encoding: None,
            xoptions: Vec::new(),
        }
    }
}

/// Environment the launcher options resolve against, when the process
/// environment is not it. Installed once at startup; absent means read
/// `std::env` directly.
static LAUNCH_ENV: LazyLock<Mutex<Option<HashMap<String, String>>>> =
    LazyLock::new(|| Mutex::new(None));

/// Supply the launcher-relevant environment explicitly, for an embedding whose
/// own process environment is not the one the program should observe.
///
/// wasm32 has no environment at all — every `std::env::var` inside the module
/// returns unset regardless of the host process — so a guest resolving `-P`,
/// `-O` or PYTHONWARNINGS from `std::env` reads them all as absent. Passing the
/// entries in restores the fold. Once installed the table is authoritative: a
/// name missing from it is unset, never a fallback to `std::env`.
pub fn set_launch_env(entries: impl IntoIterator<Item = (String, String)>) {
    *LAUNCH_ENV.lock().unwrap() = Some(entries.into_iter().collect());
}

/// The names [`finalize`] reads, for a host that has to forward them to an
/// embedding with no environment of its own. Locale variables are included
/// because an unset chain resolves to the C locale, which coerces UTF-8 mode.
pub const LAUNCH_ENV_NAMES: &[&str] = &[
    "PYTHONSAFEPATH",
    "PYTHONOPTIMIZE",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONUTF8",
    "PYTHONWARNINGS",
    "PYTHONIOENCODING",
    "LC_ALL",
    "LC_CTYPE",
    "LANG",
];

fn read(name: &str) -> Option<String> {
    if let Some(table) = LAUNCH_ENV.lock().unwrap().as_ref() {
        return table.get(name).cloned();
    }
    std::env::var(name).ok()
}

/// Presence of a variable, without decoding it. `_Py_GetEnv` tests the raw
/// bytes, so a value that is not valid Unicode still counts as set — which the
/// process-environment path preserves and an installed table cannot, its values
/// having already been decoded to cross the embedding boundary.
fn is_set_nonempty(name: &str) -> bool {
    if let Some(table) = LAUNCH_ENV.lock().unwrap().as_ref() {
        return table.get(name).is_some_and(|value| !value.is_empty());
    }
    std::env::var_os(name).is_some_and(|value| !value.is_empty())
}

fn locale_implies_utf8_mode() -> bool {
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
        if let Some(value) = read("PYTHONUTF8") {
            if !value.is_empty() {
                return match value.as_str() {
                    "0" => Ok(0),
                    "1" => Ok(1),
                    _ => Err(PreConfigError(
                        "invalid PYTHONUTF8 environment variable value",
                    )),
                };
            }
        }
    }
    Ok(if locale_implies_utf8_mode() { 1 } else { 0 })
}

/// `_Py_get_env_flag`: an integer environment flag. An unset or empty value is
/// absent; a clean non-negative integer is used as-is; anything else (trailing
/// junk, negative, overflow) counts as 1.
fn env_int_flag(name: &str) -> Option<u32> {
    let raw = read(name)?;
    if raw.is_empty() {
        return None;
    }
    let value = match raw.parse::<i64>() {
        Ok(v) if (0..=i64::from(u32::MAX)).contains(&v) => v as u32,
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
    if !flags.ignore_environment {
        if let Some(v) = env_int_flag("PYTHONOPTIMIZE") {
            level = level.max(i64::from(v));
        }
    }
    level
}

/// `-B` folded with PYTHONDONTWRITEBYTECODE: either disables bytecode caches.
fn resolve_dont_write_bytecode(flags: &LaunchFlags) -> bool {
    if flags.dont_write_bytecode {
        return true;
    }
    if !flags.ignore_environment {
        if let Some(v) = env_int_flag("PYTHONDONTWRITEBYTECODE") {
            return v != 0;
        }
    }
    false
}

/// `-P` folded with PYTHONSAFEPATH (`config_read_env_vars`). The variable is a
/// bare presence flag read through `_Py_GetEnv`, not an integer one: any
/// non-empty value enables safe path — `"0"` included — while an empty value
/// counts as unset. It only ever sets the flag, so an explicit `-P` survives
/// `-E`.
fn resolve_safe_path(flags: &LaunchFlags) -> bool {
    flags.safe_path || (!flags.ignore_environment && is_set_nonempty("PYTHONSAFEPATH"))
}

/// Fold the environment over the options a command line named. An embedding
/// without a command line passes `LaunchFlags::default()`.
pub fn finalize(mut flags: LaunchFlags) -> Result<LaunchFlags, PreConfigError> {
    flags.utf8_mode = Some(resolve_utf8_mode(&flags)?);
    flags.optimize = resolve_optimize(&flags);
    flags.dont_write_bytecode = resolve_dont_write_bytecode(&flags);
    flags.safe_path = resolve_safe_path(&flags);
    flags.stdio_encoding = if flags.ignore_environment {
        None
    } else {
        read("PYTHONIOENCODING")
    };
    // pypy/interpreter/app_main.py:892-906 — lowest-precedence entries first;
    // the warnings module installs later entries ahead of earlier ones.
    let mut warnoptions = Vec::new();
    if flags.dev_mode {
        warnoptions.push("default".to_string());
    }
    if !flags.ignore_environment {
        if let Some(value) = read("PYTHONWARNINGS") {
            if !value.is_empty() {
                warnoptions.extend(value.split(',').map(str::to_string));
            }
        }
    }
    warnoptions.append(&mut flags.warnoptions);
    if flags.bytes_warning > 0 {
        warnoptions.push(if flags.bytes_warning > 1 {
            "error::BytesWarning".to_string()
        } else {
            "default::BytesWarning".to_string()
        });
    }
    flags.warnoptions = warnoptions;
    Ok(flags)
}
