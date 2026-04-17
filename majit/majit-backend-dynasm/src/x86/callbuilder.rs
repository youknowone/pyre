//! Port of `rpython/jit/backend/x86/callbuilder.py`.
//!
//! Only the ABI argument-register tables are needed by the current
//! Rust port.  Their definitions stay here, matching upstream, and
//! are consumed by `x86/reghint.rs`.

use crate::regloc::{
    ECX, EDI, EDX, ESI, R8, R9, RegLoc, XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
};

/// callbuilder.py:519-524 `CallBuilder64.ARGUMENTS_GPR`
#[cfg(not(target_os = "windows"))]
pub const ARGUMENTS_GPR: &[RegLoc] = &[EDI, ESI, EDX, ECX, R8, R9];
#[cfg(target_os = "windows")]
pub const ARGUMENTS_GPR: &[RegLoc] = &[ECX, EDX, R8, R9];

/// callbuilder.py:519-524 `CallBuilder64.ARGUMENTS_XMM`
#[cfg(not(target_os = "windows"))]
pub const ARGUMENTS_XMM: &[RegLoc] = &[XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7];
#[cfg(target_os = "windows")]
pub const ARGUMENTS_XMM: &[RegLoc] = &[XMM0, XMM1, XMM2, XMM3];
