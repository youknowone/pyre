//! Port of `rpython/jit/backend/x86/arch.py`.
//!
//! Frame layout constants for the x86_64 backend.  pyre's
//! `dynasm/arch.rs` re-exports these so existing call sites continue
//! to use `crate::arch::*` while the per-arch source matches upstream.

/// arch.py:8 / aarch64/arch.py:1
pub const WORD: usize = 8;

/// arch.py:12
pub const IS_X86_64: bool = true;

//        +--------------------+    <== aligned to 16 bytes
//        |   return address   |
//        +--------------------+           ----------------------.
//        |    saved regs      |                FRAME_FIXED_SIZE |
//        +--------------------+       --------------------.     |
//        |   scratch          |          PASS_ON_MY_FRAME |     |
//        |      space         |                           |     |
//        +--------------------+    <== aligned to 16 -----' ----'
//
// All the rest of the data is in a GC-managed variable-size jitframe.
// This jitframe object's address is always stored in the frame
// register.  The object layout itself lives in llsupport/jitframe.py;
// arch.py only defines the managed-register prefix inside `jf_frame`.

/// arch.py:46 — rbp + rbx + r12 + r13 + r14 + r15 + threadlocal +
/// 12 extra = 19, plus 4 for vmprof.
pub const FRAME_FIXED_SIZE: usize = 19 + 4;

/// arch.py:49
pub const PASS_ON_MY_FRAME: usize = 12;

/// arch.py:50 — 13 GPR + 15 XMM
pub const JITFRAME_FIXED_SIZE: usize = 28;

/// arch.py:54 — threadlocal_addr offset in frame.
pub const THREADLOCAL_OFS: usize = (FRAME_FIXED_SIZE - 1) * WORD;

/// arch.py:65 — return address + FRAME_FIXED_SIZE words.
pub const DEFAULT_FRAME_BYTES: usize = (1 + FRAME_FIXED_SIZE) * WORD;
