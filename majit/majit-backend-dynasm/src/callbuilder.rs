//! Compatibility shim.
//!
//! Upstream keeps `CallBuilder64` and the ABI argument-register tables
//! in `rpython/jit/backend/x86/callbuilder.py`.  The real definitions
//! now live in [`crate::x86::callbuilder`]; this shared module remains
//! only as a re-export for any older call sites.

#[cfg(target_arch = "x86_64")]
pub use crate::x86::callbuilder::*;
