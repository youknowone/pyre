//! Optional builtin modules — `pypy/module/` (non-essential subset).
//!
//! PyPy classifies builtin modules into three tiers:
//!
//! | Tier       | PyPy config          | pyre location          |
//! |------------|----------------------|------------------------|
//! | Essential  | always loaded        | `pyre-interpreter`     |
//! | Default    | on by default        | `pyre-module` (here)   |
//! | Working    | opt-in               | `pyre-module` (here)   |
//!
//! Essential modules (`__builtin__`, `sys`) live in `pyre-interpreter`
//! because they are inseparable from the interpreter bootstrap.
//!
//! Everything else belongs here.  Modules will be migrated from
//! `pyre-interpreter/src/module/` as they grow.

/// Keep the `module::` path prefix so harvested hint paths and
/// `should_lower_module` stay `module::<name>` after the crate split.
pub mod module;

/// Register default/working modules into the interpreter builtin table.
///
/// The interpreter does not depend on this crate. The final binary calls
/// [`register`] before `install_builtin_modules`.
pub fn install_optional_modules() {
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("_uuid", module::_uuid::init);
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("grp", module::grp::init);
    #[cfg(all(unix, not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("_posixshmem", module::_posixshmem::init);
    #[cfg(all(unix, not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("syslog", module::syslog::init);
    pyre_interpreter::importing::register_builtin_module("_blake2", module::_blake2::init);
}

/// Install [`install_optional_modules`] as the interpreter's optional-module hook.
pub fn register() {
    pyre_interpreter::importing::set_optional_builtin_modules(install_optional_modules);
}
