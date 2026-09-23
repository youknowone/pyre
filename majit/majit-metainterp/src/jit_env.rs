//! Pyre-only JIT debug knobs.
//!
//! `PYRE_NO_JIT`, `PYRE_JIT`, `MAJIT_NO_BRIDGE`, `PYRE_NO_JD1`, `PYRE_JD1`,
//! `PYRE_JD1_NO_ENTER`, `PYRE_JD1_THRESHOLD`, `MAJIT_BRIDGE_BAIL`,
//! `MAJIT_MAX_BRIDGES`, and `MAJIT_SKIP_BRIDGES` change what the guest
//! executes. They are not `JitDriver` parameters: those are installed with
//! `set_user_param` onto `WarmEnterState.set_param_*`. These knobs stay
//! environment reads, against [`majit_ir::environ`] — the same map
//! `read_from_env` uses for `os.environ`.

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

/// Replace the one environment. Call before the first read: each site caches
/// the answer in a `OnceLock`.
pub fn set_supplied_env(entries: Vec<(String, String)>) {
    majit_ir::environ::install(string_entries(entries));
}

/// Replace the one environment with raw values. The launcher needs bytes:
/// `PYTHONSAFEPATH` is set by any nonempty value, decoded or not.
pub fn install(entries: Vec<(String, Vec<u8>)>) {
    majit_ir::environ::install(entries);
}

/// Upsert into the one environment without dropping names another blob wrote.
pub fn extend(entries: Vec<(String, Vec<u8>)>) {
    majit_ir::environ::extend(entries);
}

pub fn upsert(name: &str, value: Vec<u8>) {
    majit_ir::environ::upsert(name, value);
}

pub fn remove(name: &str) {
    majit_ir::environ::remove(name);
}

pub fn entries() -> Vec<(String, Vec<u8>)> {
    majit_ir::environ::entries()
}

fn string_entries(entries: Vec<(String, String)>) -> Vec<(String, Vec<u8>)> {
    entries
        .into_iter()
        .map(|(name, value)| (name, value.into_bytes()))
        .collect()
}

/// `std::env::var_os`, then the one installed environment.
pub fn env_var_os(varname: &str) -> Option<OsString> {
    majit_ir::environ::env_var_os(varname)
}

/// `std::env::var`, then the one installed environment.
///
/// The value form, for the knobs read as `== "0"` rather than by presence.
pub fn env_var(varname: &str) -> Option<String> {
    majit_ir::environ::env_var(varname)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One install has to be visible to a collector knob and a JIT knob.
    /// Separate `SUPPLIED_ENV` tables fail this: the setter that wrote the
    /// value is not the reader that consumes it.
    #[test]
    fn one_environment_is_visible_to_a_gc_knob_and_a_jit_knob() {
        unsafe {
            std::env::remove_var("PYPY_GC_DEBUG");
            std::env::remove_var("MAJIT_NO_BRIDGE");
        }
        majit_gc::collector::set_supplied_env(vec![(
            "MAJIT_NO_BRIDGE".to_string(),
            "1".to_string(),
        )]);
        assert!(
            env_var_os("MAJIT_NO_BRIDGE").is_some(),
            "a name installed for the collector must be the name the jit knob reads"
        );
        set_supplied_env(vec![("PYPY_GC_DEBUG".to_string(), "2".to_string())]);
        assert_eq!(
            majit_gc::collector::GcConfig::default().debug,
            2,
            "a name installed for the jit must be the name the collector reads"
        );
        let installed = entries();
        assert_eq!(installed.len(), 1, "one install replaced the one map");
        assert_eq!(installed[0].0, "PYPY_GC_DEBUG");
        set_supplied_env(Vec::new());
        majit_gc::collector::set_supplied_env(Vec::new());
    }
}
