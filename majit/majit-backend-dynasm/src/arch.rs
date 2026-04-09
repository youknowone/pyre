/// arch.py: Constants that depend on whether we are on 32-bit or 64-bit.
///
/// The frame is absolutely standard. Stores callee-saved registers,
/// return address and some scratch space for arguments.

/// arch.py:8 / aarch64/arch.py:1
pub const WORD: usize = 8;
/// arch.py:12
#[cfg(target_arch = "x86_64")]
pub const IS_X86_64: bool = true;
#[cfg(target_arch = "aarch64")]
pub const IS_X86_64: bool = false;

//        +--------------------+    <== aligned to 16 bytes
//        |   return address   |
//        +--------------------+           ----------------------.
//        |    saved regs      |                FRAME_FIXED_SIZE |
//        +--------------------+       --------------------.     |
//        |   scratch          |          PASS_ON_MY_FRAME |     |
//        |      space         |                           |     |
//        +--------------------+    <== aligned to 16 -----' ----'

// All the rest of the data is in a GC-managed variable-size jitframe.
// This jitframe object's address is always stored in the frame register.
// The object layout itself lives in llsupport/jitframe.py; arch.py only
// defines the managed-register prefix inside `jf_frame`.

/// arch.py:46 — rbp + rbx + r12 + r13 + r14 + r15 + threadlocal + 12 extra = 19
#[cfg(target_arch = "x86_64")]
pub const FRAME_FIXED_SIZE: usize = 19 + 4; // 4 for vmprof
#[cfg(target_arch = "aarch64")]
pub const FRAME_FIXED_SIZE: usize = 0;

/// arch.py:49
#[cfg(target_arch = "x86_64")]
pub const PASS_ON_MY_FRAME: usize = 12;
#[cfg(target_arch = "aarch64")]
pub const PASS_ON_MY_FRAME: usize = 0;

/// arch.py:50 — 13 GPR + 15 XMM
#[cfg(target_arch = "x86_64")]
pub const JITFRAME_FIXED_SIZE: usize = 28;
/// aarch64/arch.py:13 — NUM_MANAGED_REGS + NUM_VFP_REGS
#[cfg(target_arch = "aarch64")]
pub const JITFRAME_FIXED_SIZE: usize = 24;

/// arch.py:54 — threadlocal_addr offset in frame
#[cfg(target_arch = "x86_64")]
pub const THREADLOCAL_OFS: usize = (FRAME_FIXED_SIZE - 1) * WORD;
#[cfg(target_arch = "aarch64")]
pub const THREADLOCAL_OFS: usize = 0;

/// arch.py:65 — return address + FRAME_FIXED_SIZE words
#[cfg(target_arch = "x86_64")]
pub const DEFAULT_FRAME_BYTES: usize = (1 + FRAME_FIXED_SIZE) * WORD;
#[cfg(target_arch = "aarch64")]
pub const DEFAULT_FRAME_BYTES: usize = 0;
