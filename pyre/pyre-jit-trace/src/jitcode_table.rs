//! Per-opcode JitCode table — runtime loader for the artifact emitted
//! by `build.rs`.
//!
//! Phase C of the eval-loop automation plan. The build script runs
//! `majit-codewriter` over `pyre-interpreter` source, calls the assembler
//! on each opcode arm's flattened SSARepr, and writes the resulting
//! `JitCode` payloads to `OUT_DIR/pyre_opcode_jitcodes.json`. This module
//! `include_str!`s that JSON and lazily deserializes it on first access.
//!
//! No runtime consumers yet — Phase D will plumb `MIFrame` to dispatch
//! through `BlackholeInterpreter::dispatch_one()` using these payloads.
//!
//! RPython equivalent: at translation time, `rpython/jit/codewriter/`
//! produces a `JitCode` per portal/callee graph. majit produces a JitCode
//! per Python opcode arm (matching pyjitpl.MIFrame's per-opcode dispatch
//! at trace time).

use std::sync::OnceLock;

use majit_codewriter::assembler::JitCode;
use serde::Deserialize;

/// Embedded JSON payload — pyre's per-opcode jitcodes, produced by
/// `pyre/pyre-jit-trace/build.rs`.
const OPCODE_JITCODES_JSON: &str =
    include_str!(concat!(env!("OUT_DIR"), "/pyre_opcode_jitcodes.json"));

#[derive(Debug, Clone, Deserialize)]
pub struct OpcodeJitCodeEntry {
    /// `OpcodeDispatchSelector::canonical_key()` — e.g. `"Instruction::Add"`
    /// or `"Instruction::ExtendedArg | Instruction::Resume"`.
    pub selector: String,
    /// Pre-assembled jitcode bytes + register/constant metadata.
    pub jitcode: JitCode,
}

/// Lazily deserialize the embedded JSON on first access.
fn entries() -> &'static Vec<OpcodeJitCodeEntry> {
    static CACHE: OnceLock<Vec<OpcodeJitCodeEntry>> = OnceLock::new();
    CACHE.get_or_init(|| {
        serde_json::from_str(OPCODE_JITCODES_JSON).unwrap_or_else(|e| {
            panic!("pyre_opcode_jitcodes.json deserialize: {e}");
        })
    })
}

/// Number of opcode arms with assembled jitcode.
pub fn len() -> usize {
    entries().len()
}

/// Look up a jitcode by selector key (e.g. `"Instruction::Add"`).
///
/// Linear scan — fine for one-time lookups during MIFrame init.
/// Phase D may add a `HashMap<String, usize>` index if needed.
pub fn find(selector_key: &str) -> Option<&'static OpcodeJitCodeEntry> {
    entries().iter().find(|e| e.selector == selector_key)
}

/// Iterate all entries (Phase D will use this to build a dispatch table
/// keyed by `Instruction` enum index).
pub fn iter() -> impl Iterator<Item = &'static OpcodeJitCodeEntry> {
    entries().iter()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_some_entries() {
        // Phase C contract: build.rs assembled all 108 opcode arms.
        let n = len();
        assert!(n > 0, "expected at least one opcode jitcode entry, got {n}");
    }

    #[test]
    fn entries_have_nonempty_code() {
        for entry in iter() {
            assert!(
                !entry.jitcode.code.is_empty(),
                "selector {:?} has empty jitcode bytes",
                entry.selector
            );
        }
    }
}
