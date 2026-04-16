//! Port of `rpython/jit/backend/aarch64/arch.py`.
//!
//! Frame layout constants for the aarch64 backend.  pyre's
//! `dynasm/arch.rs` re-exports these so existing call sites continue
//! to use `crate::arch::*` while the per-arch source matches upstream.

/// aarch64/arch.py:1
pub const WORD: usize = 8;

/// pyre exposes the same flag the x86 file uses; the value is `false`
/// here so consumers can write `if IS_X86_64 { ... }` without an
/// explicit `cfg!`.
pub const IS_X86_64: bool = false;

/// aarch64/arch.py: stack-frame fixed header is empty under pyre's
/// AAPCS64 prologue (callee-save registers go into the standard
/// register save area emitted by aarch64/assembler.rs `_call_header`).
pub const FRAME_FIXED_SIZE: usize = 0;

/// aarch64/arch.py: no scratch slots reserved on the stack.
pub const PASS_ON_MY_FRAME: usize = 0;

/// aarch64/arch.py:13 — `JITFRAME_FIXED_SIZE = NUM_MANAGED_REGS +
/// NUM_VFP_REGS`.
///
/// pyre grows NUM_MANAGED_REGS from upstream's 16 to 18 to activate
/// aarch64/registers.py:14's commented `, x21, x22]` extension (see
/// aarch64/regalloc.rs `all_core_regs`).  Values: 18 GPR + 8 VFP = 26.
pub const JITFRAME_FIXED_SIZE: usize = 26;

/// aarch64/arch.py: thread-local data is reached via the system TLS
/// register, not via a fixed slot in the JIT frame.
pub const THREADLOCAL_OFS: usize = 0;

/// aarch64/arch.py: no per-frame default-bytes reservation.
pub const DEFAULT_FRAME_BYTES: usize = 0;
