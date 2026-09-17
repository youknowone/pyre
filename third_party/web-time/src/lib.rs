//! Stand-in for the crates.io `web-time` crate that does not pull wasm-bindgen.
//!
//! `rustpython-host_env` enables rustls-pki-types `web` on wasm32-unknown-unknown
//! so `UnixTime::now` exists for Charon. The published `web-time` implements
//! that clock through `Date.now()` and therefore imports
//! `__wbindgen_placeholder__`, which a wasmtime host cannot satisfy.
//! Interpreter `_ssl` is not compiled into the wasm guest, so a clock that
//! reports the Unix epoch is enough for the type to exist.

pub use core::time::Duration;

/// Same shape as `web_time::SystemTime`; rustls-pki-types only calls `now`,
/// `UNIX_EPOCH`, and `duration_since`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SystemTime(Duration);

/// Same shape as `web_time::Instant`. Unused by rustls-pki-types.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Instant(Duration);

impl SystemTime {
    pub const UNIX_EPOCH: Self = Self(Duration::ZERO);

    #[must_use]
    pub fn now() -> Self {
        Self::UNIX_EPOCH
    }

    pub fn duration_since(&self, earlier: Self) -> Result<Duration, SystemTimeError> {
        self.0
            .checked_sub(earlier.0)
            .ok_or(SystemTimeError(Duration::ZERO))
    }
}

impl Instant {
    #[must_use]
    pub fn now() -> Self {
        Self(Duration::ZERO)
    }

    #[must_use]
    pub fn duration_since(&self, earlier: Self) -> Duration {
        self.0.saturating_sub(earlier.0)
    }
}

/// Error from [`SystemTime::duration_since`].
#[derive(Clone, Debug)]
pub struct SystemTimeError(Duration);

impl SystemTimeError {
    #[must_use]
    pub fn duration(&self) -> Duration {
        self.0
    }
}
