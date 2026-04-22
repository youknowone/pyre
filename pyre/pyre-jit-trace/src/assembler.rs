//! Pyre's lazy-codewriter analog of RPython's
//! `rpython.jit.codewriter.assembler.Assembler`.
//!
//! Upstream's `Assembler` owns the writer-side codewriter state while
//! `MetaInterpStaticData` is the reader-side snapshot. The split matters:
//! the dedup dict `all_liveness_positions`, the running `all_liveness`
//! buffer, and the `insns` name→opcode table all live on the assembler
//! because they mutate during codewriter passes; staticdata consumes
//! a frozen snapshot via `finish_setup`.
//!
//! pyre has no AOT codewriter pass — JitCodes are compiled lazily from
//! CPython bytecode the first time a code object runs under the JIT. This
//! module keeps the writer-side liveness state separate from staticdata, but
//! the opcode table is still majit's fixed adapter table rather than
//! RPython's dense `Assembler.write_insn()`-grown dict.
//!
//! Reference: `rpython/jit/codewriter/assembler.py:19-32`,
//! `rpython/jit/codewriter/assembler.py:234-248` (`_encode_liveness`),
//! `rpython/jit/metainterp/pyjitpl.py:2264`
//! (`self.liveness_info = "".join(asm.all_liveness)`).

use std::cell::RefCell;
use std::collections::HashMap;

/// RPython `assembler.py:19-32` `Assembler.__init__`.
///
/// Only the fields pyre actually consumes are carried here:
/// - `insns`           — opcode name ("live/", "catch_exception/L", …)
///   mapped to opcode number; source for `MetaInterpStaticData.setup_insns`.
/// - `all_liveness`    — running byte buffer appended by `_encode_liveness`.
/// - `all_liveness_length` — explicit length counter mirroring upstream.
/// - `all_liveness_positions` — dedup dict from bitset key to offset.
/// - `num_liveness_ops` — running count of liveness writes (diagnostic).
pub struct AssemblerState {
    pub insns: HashMap<String, u8>,
    pub all_liveness: Vec<u8>,
    pub all_liveness_length: usize,
    pub all_liveness_positions: HashMap<(Vec<u8>, Vec<u8>, Vec<u8>), u16>,
    pub num_liveness_ops: usize,
}

impl AssemblerState {
    fn new() -> Self {
        Self {
            insns: HashMap::new(),
            all_liveness: Vec::new(),
            all_liveness_length: 0,
            all_liveness_positions: HashMap::new(),
            num_liveness_ops: 0,
        }
    }
}

thread_local! {
    /// pyre-local analog of the `Assembler` instance held by RPython's
    /// `CodeWriter`. Accessed from `pyre_jit_trace::state::intern_liveness`
    /// and from the first-use initializer that wires `setup_insns`.
    ///
    /// `rpython/jit/codewriter/codewriter.py:20-23` puts the Assembler
    /// on the CodeWriter. pyre would normally do the same, but the
    /// reader side (blackhole / resume in `pyre_jit_trace`) cannot
    /// depend on `pyre_jit`, so the single authoritative Assembler
    /// state lives in this lower crate and the `pyre_jit` side exposes
    /// it through its own `Assembler` handle. `all_liveness` /
    /// `num_liveness_ops` accumulate here across every jitcode
    /// compiled on this thread — that is the canonical analog of
    /// `codewriter.assembler.all_liveness`.
    pub(crate) static ASSEMBLER_STATE: RefCell<AssemblerState> =
        RefCell::new(AssemblerState::new());
}

/// Snapshot the current thread's `AssemblerState.all_liveness`.
///
/// `rpython/jit/codewriter/assembler.py:29` exposes `all_liveness` as
/// a public attribute on the `Assembler` so `pyjitpl.py:2264` can
/// publish it via `self.liveness_info = "".join(asm.all_liveness)`.
/// Returns a copy because the thread-local borrow cannot escape the
/// `LocalKey::with` closure.
pub fn all_liveness_snapshot() -> Vec<u8> {
    ASSEMBLER_STATE.with(|r| r.borrow().all_liveness.clone())
}

/// Snapshot the current thread's `AssemblerState.all_liveness_length`.
/// Mirrors `rpython/jit/codewriter/assembler.py:30`.
pub fn all_liveness_length() -> usize {
    ASSEMBLER_STATE.with(|r| r.borrow().all_liveness_length)
}

/// Snapshot the current thread's `AssemblerState.num_liveness_ops`.
/// Mirrors `rpython/jit/codewriter/assembler.py:32` — incremented
/// once per `-live-` insn during assembly.
pub fn num_liveness_ops() -> usize {
    ASSEMBLER_STATE.with(|r| r.borrow().num_liveness_ops)
}

/// Publish the writer-side `Assembler`'s latest snapshot to the
/// blackhole-reader thread-local. `pyre_jit::Assembler` owns the
/// canonical per-instance state (line-by-line with
/// `rpython/jit/codewriter/assembler.py:19-32`); this is only the
/// pyre-layering bridge so the blackhole in this lower crate can read
/// without a circular dep on `pyre_jit`. Not a second source of truth
/// — every publish replaces the mirror entirely.
pub fn publish_state(
    insns: &HashMap<String, u8>,
    all_liveness: &[u8],
    all_liveness_length: usize,
    num_liveness_ops: usize,
) {
    ASSEMBLER_STATE.with(|r| {
        let mut asm = r.borrow_mut();
        asm.insns = insns.clone();
        asm.all_liveness.clear();
        asm.all_liveness.extend_from_slice(all_liveness);
        asm.all_liveness_length = all_liveness_length;
        asm.num_liveness_ops = num_liveness_ops;
    });
    crate::state::publish_liveness_info(all_liveness.to_vec());
}
