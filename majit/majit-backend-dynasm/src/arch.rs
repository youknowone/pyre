/// arch.py: Constants that depend on whether we are on 32-bit or 64-bit.
///
/// The frame is absolutely standard. Stores callee-saved registers,
/// return address and some scratch space for arguments.

/// arch.py:8
pub const WORD: usize = 8; // x86_64 only
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

// All the rest of the data is in a GC-managed variable-size "frame".
// This frame object's address is always stored in the register RBP.
// A frame is jit.backend.llsupport.llmodel.JITFRAME = GcArray(Signed).

/// arch.py:46 — rbp + rbx + r12 + r13 + r14 + r15 + threadlocal + 12 extra = 19
pub const FRAME_FIXED_SIZE: usize = 19 + 4; // 4 for vmprof

/// arch.py:49
pub const PASS_ON_MY_FRAME: usize = 12;

/// arch.py:50 — 13 GPR + 15 XMM
pub const JITFRAME_FIXED_SIZE: usize = 28;

/// arch.py:54 — threadlocal_addr offset in frame
pub const THREADLOCAL_OFS: usize = (FRAME_FIXED_SIZE - 1) * WORD;

/// arch.py:65 — return address + FRAME_FIXED_SIZE words
pub const DEFAULT_FRAME_BYTES: usize = (1 + FRAME_FIXED_SIZE) * WORD;

// aarch64 arch constants (when target is aarch64)
// TODO: read from rpython/jit/backend/aarch64/arch.py

/// jitframe.py: offset of jf_descr in JITFRAME (in WORD units).
/// RPython: JITFRAME header layout is fixed by GC.
pub const JF_DESCR_OFS: usize = 0;

/// jitframe.py: offset of first frame item (jf_frame[0]).
pub const JF_FRAME_ITEM0_OFS: usize = 1;
