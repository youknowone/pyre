//! The one environment a guest reads.
//!
//! `read_from_env` reads `os.environ` and nothing else. Natively that map is
//! the process environment. A wasm guest has no process environment, so the
//! host installs this map once through [`install`] before anything reads it.
//! `majit-gc`, `majit-metainterp` and the launcher all come here; there is no
//! second table.

use parking_lot::RwLock;
use std::ffi::OsString;

static ENV: RwLock<Vec<(String, Vec<u8>)>> = RwLock::new(Vec::new());

/// Replace the installed map. The host calls this once, before any read.
pub fn install(entries: Vec<(String, Vec<u8>)>) {
    *ENV.write() = entries;
}

/// Upsert entries into the installed map. A later record for the same name
/// replaces the earlier one. Used when an embedder still hands the map over
/// in more than one blob; both blobs land in [`ENV`].
pub fn extend(entries: impl IntoIterator<Item = (String, Vec<u8>)>) {
    let mut env = ENV.write();
    for (name, value) in entries {
        if let Some((_, slot)) = env.iter_mut().find(|(existing, _)| *existing == name) {
            *slot = value;
        } else {
            env.push((name, value));
        }
    }
}

pub fn upsert(name: &str, value: Vec<u8>) {
    extend([(name.to_string(), value)]);
}

pub fn remove(name: &str) {
    ENV.write().retain(|(existing, _)| existing != name);
}

pub fn entries() -> Vec<(String, Vec<u8>)> {
    ENV.read().clone()
}

fn supplied(name: &str) -> Option<Vec<u8>> {
    ENV.read()
        .iter()
        .find(|(existing, _)| existing == name)
        .map(|(_, value)| value.clone())
}

/// `std::env::var`, then the installed map. A non-UTF-8 value is absent, the
/// same as `std::env::var`'s `VarError::NotUnicode`.
pub fn env_var(name: &str) -> Option<String> {
    if let Ok(value) = std::env::var(name) {
        return Some(value);
    }
    supplied(name).and_then(|bytes| String::from_utf8(bytes).ok())
}

/// `std::env::var_os`, then the installed map.
pub fn env_var_os(name: &str) -> Option<OsString> {
    if let Some(value) = std::env::var_os(name) {
        return Some(value);
    }
    supplied(name).map(os_string_from_bytes)
}

/// Presence, matching `std::env::var_os(name).is_some()`: an empty value counts.
pub fn env_is_set(name: &str) -> bool {
    std::env::var_os(name).is_some() || ENV.read().iter().any(|(existing, _)| existing == name)
}

fn os_string_from_bytes(bytes: Vec<u8>) -> OsString {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        OsString::from_vec(bytes)
    }
    #[cfg(not(unix))]
    {
        OsString::from(String::from_utf8_lossy(&bytes).into_owned())
    }
}
