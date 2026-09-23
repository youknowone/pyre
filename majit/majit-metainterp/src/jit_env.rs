//! JIT knobs an embedder supplies because the platform gives the process no
//! environment to read them out of.
//!
//! `PYRE_NO_JIT`, `PYRE_JIT`, `MAJIT_NO_BRIDGE`, `PYRE_NO_JD1`, `PYRE_JD1`,
//! `PYRE_JD1_NO_ENTER`, `PYRE_JD1_THRESHOLD`, `MAJIT_BRIDGE_BAIL`,
//! `MAJIT_MAX_BRIDGES`, and `MAJIT_SKIP_BRIDGES` change
//! what the guest executes, not merely what it prints. `wasm32-unknown-unknown`
//! has a permanently empty `std::env`, so without a supplied table those knobs
//! read as unset no matter what the host was configured with.

use parking_lot::RwLock;
use std::ffi::OsString;

/// The variables the JIT knobs resolve against, for an embedder that has to
/// hand its environment over rather than share it.
///
/// Published here so such a host does not keep its own copy of the list in step
/// with the read sites.
pub const JIT_ENV_NAMES: &[&str] = &[
    "PYRE_NO_JIT",
    "PYRE_JIT",
    "MAJIT_NO_BRIDGE",
    "PYRE_NO_JD1",
    "PYRE_JD1",
    "PYRE_JD1_NO_ENTER",
    "PYRE_JD1_THRESHOLD",
    "MAJIT_BRIDGE_BAIL",
    "MAJIT_MAX_BRIDGES",
    "MAJIT_SKIP_BRIDGES",
];

/// Environment an embedder supplies because the platform gives the process
/// none. Read only where `std::env` misses, so a host that has a real
/// environment resolves against it exactly as before.
///
/// `wasm32-unknown-unknown` is the case that needs it: `std::env::var_os`
/// there always returns `None`, so every name in [`JIT_ENV_NAMES`] reads as
/// unset and a guest runs with those knobs off no matter what its host was
/// configured with. The collector's `PYPY_GC_*` block and the interpreter's
/// launcher options have the same problem and the same answer.
static SUPPLIED_ENV: RwLock<Vec<(String, String)>> = RwLock::new(Vec::new());

/// Install the environment [`JIT_ENV_NAMES`] resolves against when the process
/// has none. Call before the first read: each site caches the answer in a
/// `OnceLock`.
pub fn set_supplied_env(entries: Vec<(String, String)>) {
    *SUPPLIED_ENV.write() = entries;
}

/// `std::env::var_os`, falling back to what the embedder supplied.
pub fn env_var_os(varname: &str) -> Option<OsString> {
    if let Some(value) = std::env::var_os(varname) {
        return Some(value);
    }
    let supplied = SUPPLIED_ENV.read();
    supplied
        .iter()
        .find(|(name, _)| name == varname)
        .map(|(_, value)| OsString::from(value.as_str()))
}

/// `std::env::var`, falling back to what the embedder supplied.
///
/// The value form, for the knobs read as `== "0"` rather than by presence.
pub fn env_var(varname: &str) -> Option<String> {
    if let Ok(value) = std::env::var(varname) {
        return Some(value);
    }
    let supplied = SUPPLIED_ENV.read();
    supplied
        .iter()
        .find(|(name, _)| name == varname)
        .map(|(_, value)| value.clone())
}
