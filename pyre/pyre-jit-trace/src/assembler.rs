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

use indexmap::IndexMap;

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
    pub insns: IndexMap<String, u8>,
    pub all_liveness: Vec<u8>,
    pub all_liveness_length: usize,
    pub all_liveness_positions: IndexMap<(Vec<u8>, Vec<u8>, Vec<u8>), u16>,
    pub num_liveness_ops: usize,
}

impl AssemblerState {
    fn new() -> Self {
        // `assembler.py:29-31` gives one `Assembler` one `all_liveness`
        // buffer, and `codewriter.py:73-86 make_jitcodes` drains *every*
        // pending graph through it, so upstream has exactly one offset space
        // for the whole program. pyre splits the drain across two processes —
        // `build.rs` assembles the extracted interpreter graphs, the runtime
        // codewriter assembles Python-bytecode graphs — but the offsets baked
        // into the build-time jitcodes are positions in that same buffer.
        // Resuming from the build-time bytes is what makes it one buffer
        // again: the baked offsets stay valid and the runtime's own
        // `_encode_liveness` allocates strictly above them.
        //
        // `pyre_jit::Assembler::resuming_build_time_liveness` seeds the
        // writer side the same way, so a `publish_state` wholesale replace
        // never rewinds past this prefix.
        //
        // `assembler.py:20 self.insns = {}` continues the same way. The
        // build-time jitcodes' `-live-` markers carry the canonical opcode
        // byte, and `blackhole.py:55-61` recovers it as `asm.insns['live/']`;
        // starting empty leaves `MetaInterpStaticData.op_live` at its unset
        // sentinel, and `can_decode_live_vars` then hunts for that sentinel as
        // a marker byte and declines every build-time resume.
        let all_liveness = crate::jitcode_runtime::all_liveness().to_vec();
        let all_liveness_length = all_liveness.len();
        let insns = crate::jitcode_runtime::insns_opname_to_byte().clone();
        Self {
            insns,
            all_liveness,
            all_liveness_length,
            all_liveness_positions: IndexMap::new(),
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
    insns: &IndexMap<String, u8>,
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
        // The dedup table is keyed by offsets into the previous
        // all_liveness buffer; clear it so subsequent intern_liveness
        // lookups cannot reuse stale offsets against the fresh buffer.
        asm.all_liveness_positions.clear();
    });
    crate::state::publish_liveness_info(all_liveness.to_vec());
}

#[cfg(test)]
mod tests {
    /// `assembler.py:29-31` — one `Assembler`, one `all_liveness` offset
    /// space. The build-time drain's bytes are the prefix of that buffer, so
    /// every offset baked into a build-time jitcode's `-live-` operand
    /// addresses the same byte in `metainterp_sd.liveness_info` that
    /// `resume.py:1022` reads. Losing the prefix would not fail loudly — a
    /// baked offset would land inside an unrelated runtime triple and hand
    /// back a mistyped value count — so pin it.
    #[test]
    fn assembler_state_resumes_the_build_time_liveness_prefix() {
        let build_time = crate::jitcode_runtime::all_liveness();
        assert!(
            !build_time.is_empty(),
            "the build-time drain produced no liveness bytes; \
             `liveness.bin` is empty or failed to deserialize"
        );
        super::ASSEMBLER_STATE.with(|r| {
            let asm = r.borrow();
            assert_eq!(
                asm.all_liveness_length,
                asm.all_liveness.len(),
                "`assembler.py:30 all_liveness_length` must track the buffer"
            );
            assert!(
                asm.all_liveness.starts_with(build_time),
                "AssemblerState::new must resume the build-time buffer \
                 (len {} vs build-time len {})",
                asm.all_liveness.len(),
                build_time.len(),
            );
        });
    }
}
