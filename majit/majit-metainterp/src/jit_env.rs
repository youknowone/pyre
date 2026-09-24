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

/// Upsert into the one environment. Call before the first read: each site
/// caches the answer in a `OnceLock`. A later call keeps names another setter
/// already wrote. [`install`] is what replaces the map.
pub fn set_supplied_env(entries: Vec<(String, String)>) {
    majit_ir::environ::extend(string_entries(entries));
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

    /// One map has to be visible to a collector reader and a JIT reader.
    /// Separate tables fail this: the setter that wrote the value is not the
    /// reader that consumes it. The names are unique to this test, so a
    /// process variable cannot hide the installed value and nothing here
    /// writes the process environment.
    #[test]
    fn one_environment_is_visible_to_a_gc_knob_and_a_jit_knob() {
        const JIT_NAME: &str = "MAJIT_TEST_ONE_ENV_JIT_READER";
        const GC_NAME: &str = "MAJIT_TEST_ONE_ENV_GC_READER";
        majit_ir::environ::install(Vec::new());
        majit_gc::collector::set_supplied_env(vec![(JIT_NAME.to_string(), "1".to_string())]);
        assert!(
            env_var_os(JIT_NAME).is_some(),
            "a name installed for the collector must be the name the jit knob reads"
        );
        set_supplied_env(vec![(GC_NAME.to_string(), "2".to_string())]);
        assert_eq!(
            majit_ir::environ::env_var(GC_NAME).as_deref(),
            Some("2"),
            "a name installed for the jit must be the name the collector reads"
        );
        let installed = entries();
        assert_eq!(installed.len(), 2, "each setter extends the one map");
        assert!(installed.iter().any(|(name, _)| name == JIT_NAME));
        assert!(
            installed
                .iter()
                .any(|(name, value)| name == GC_NAME && value == b"2")
        );
        majit_ir::environ::install(Vec::new());
    }
}
