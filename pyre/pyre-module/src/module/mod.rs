//! Builtin modules that are not part of interpreter bootstrap.

#[cfg(all(unix, not(feature = "sandbox")))]
pub mod syslog;
