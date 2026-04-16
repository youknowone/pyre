//! Re-export shim for the per-arch `arch.rs` files (mirroring
//! `rpython/jit/backend/x86/arch.py` and `rpython/jit/backend/
//! aarch64/arch.py`).
//!
//! Upstream RPython has no `llsupport/arch.py`; the constants live
//! per-arch.  pyre keeps a thin shared module so existing call sites
//! can `use crate::arch::*` without caring about the active backend,
//! but the values themselves are owned by the per-arch source.

#[cfg(target_arch = "x86_64")]
pub use crate::x86::arch::*;

#[cfg(target_arch = "aarch64")]
pub use crate::aarch64::arch::*;
