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
    pub insns: HashMap<&'static str, u8>,
    pub all_liveness: Vec<u8>,
    pub all_liveness_length: usize,
    pub all_liveness_positions: HashMap<(Vec<u8>, Vec<u8>, Vec<u8>), u16>,
    pub num_liveness_ops: usize,
}

impl AssemblerState {
    fn new() -> Self {
        // pyre's opcode table is compile-time fixed in majit-metainterp.
        // This is an adapter input to `setup_insns`, not a full RPython
        // `asm.insns` table grown in emission order.
        Self {
            insns: majit_metainterp::jitcode::wellknown_bh_insns(),
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
    pub(crate) static ASSEMBLER_STATE: RefCell<AssemblerState> =
        RefCell::new(AssemblerState::new());
}
