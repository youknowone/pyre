//! Builtin modules that are not part of interpreter bootstrap.

#[allow(non_snake_case)]
pub mod _blake2;
#[allow(non_snake_case)]
#[cfg(all(unix, not(feature = "sandbox")))]
pub mod _posixshmem;
#[allow(non_snake_case)]
#[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
pub mod _uuid;
#[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
pub mod grp;
#[cfg(all(unix, not(feature = "sandbox")))]
pub mod syslog;
