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

use std::collections::HashMap;
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
    /// or `"Instruction::ExtendedArg | Instruction::Resume"` for multi-arm
    /// match patterns. Selectors with `|` cover multiple `Instruction`
    /// variants that share an opcode handler body.
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

/// Reverse index: variant name → entry index.
///
/// Multi-pattern arms (e.g. `"Instruction::ExtendedArg | Instruction::Resume"`)
/// register *each* variant under its own key. Phase D's MIFrame jitcode
/// dispatch will use this to look up the JitCode for a given Python
/// `Instruction` discriminant in O(1).
fn index() -> &'static HashMap<String, usize> {
    static CACHE: OnceLock<HashMap<String, usize>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let mut idx = HashMap::new();
        for (i, entry) in entries().iter().enumerate() {
            for variant in entry.selector.split('|') {
                let key = variant.trim().to_string();
                if !key.is_empty() {
                    idx.insert(key, i);
                }
            }
        }
        idx
    })
}

/// Number of opcode arms with assembled jitcode.
pub fn len() -> usize {
    entries().len()
}

/// Look up a jitcode by full selector key, including the multi-pattern form.
/// E.g. `"Instruction::ExtendedArg | Instruction::Resume"`.
pub fn find(selector_key: &str) -> Option<&'static OpcodeJitCodeEntry> {
    entries().iter().find(|e| e.selector == selector_key)
}

/// Look up a jitcode by single `Instruction` variant name, e.g.
/// `"Instruction::Add"`. Multi-pattern arms expose each of their
/// constituent variants here independently.
pub fn find_by_variant(variant_key: &str) -> Option<&'static OpcodeJitCodeEntry> {
    let i = *index().get(variant_key)?;
    Some(&entries()[i])
}

/// Iterate all entries.
pub fn iter() -> impl Iterator<Item = &'static OpcodeJitCodeEntry> {
    entries().iter()
}

/// Validate that every entry carries non-trivial jitcode and that the
/// reverse index reaches every variant. Called once at JIT startup —
/// surfaces build-pipeline regressions before they hit the trace path.
pub fn validate() -> Result<usize, String> {
    let es = entries();
    if es.is_empty() {
        return Err("pyre_opcode_jitcodes.json is empty".to_string());
    }
    let mut total_bytes = 0usize;
    for (i, e) in es.iter().enumerate() {
        if e.jitcode.code.is_empty() {
            return Err(format!(
                "entry {i} ({:?}) has empty jitcode bytes",
                e.selector
            ));
        }
        total_bytes += e.jitcode.code.len();
    }
    let idx = index();
    if idx.is_empty() {
        return Err("variant index is empty".to_string());
    }
    Ok(total_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_some_entries() {
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

    #[test]
    fn variant_index_round_trips() {
        // For every variant in the index, find_by_variant returns an entry
        // whose selector contains that variant.
        for entry in iter() {
            for variant in entry.selector.split('|') {
                let key = variant.trim();
                if key.is_empty() {
                    continue;
                }
                let found = find_by_variant(key)
                    .unwrap_or_else(|| panic!("variant {key:?} missing from index"));
                assert!(
                    found.selector.contains(key),
                    "find_by_variant({key:?}) returned wrong selector {:?}",
                    found.selector,
                );
            }
        }
    }

    #[test]
    fn validate_succeeds_at_startup() {
        let total = validate().expect("jitcode_table validation failed");
        assert!(total > 0, "validate() reported zero total bytes");
    }
}
