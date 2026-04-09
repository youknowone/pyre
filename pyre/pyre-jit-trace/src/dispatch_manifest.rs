//! Runtime opcode-dispatch manifest loader (Phase C v2).
//!
//! `pyre/pyre-jit-trace/build.rs` registers each `execute_opcode_step` match
//! arm body as a synthetic graph in `CallControl`, runs them through the
//! orthodox `CodeWriter::drain_pending_graphs` loop, and writes a manifest
//! mapping each `Instruction::*` variant to the resulting `jitcode_index`
//! into `OUT_DIR/pyre_opcode_dispatch_manifest.json`.
//!
//! This module `include_str!`s that JSON, lazily deserializes it on first
//! access, and exposes an O(1) `variant → jitcode_index` lookup. There is
//! no consumer wired up yet — Phase D will plumb `MIFrame::dispatch_jitcode`
//! through it.
//!
//! RPython parity (`rpython/jit/codewriter/codewriter.py:74-89`): RPython
//! does not have a manifest layer because PyPy's interpreter naturally has
//! one Python method per opcode and the meta-interp dispatches via Python
//! object identity in `CallControl.jitcodes[graph]`. pyre dispatches inside
//! one big match, so the manifest is a thin variant→index index laid on
//! top of the same indexed `all_jitcodes[]` invariant.

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::Deserialize;

/// Embedded JSON payload — pyre's per-variant dispatch manifest, produced by
/// `pyre/pyre-jit-trace/build.rs`.
const OPCODE_DISPATCH_MANIFEST_JSON: &str = include_str!(concat!(
    env!("OUT_DIR"),
    "/pyre_opcode_dispatch_manifest.json"
));

/// One entry in the runtime manifest.
///
/// Mirrors `OpcodeDispatchManifestEntry` in `build.rs`. The runtime side
/// stores `variant` and `jitcode_index` for fast lookup; `arm_id`,
/// `selector`, and `debug_name` are kept for diagnostics / `validate()`.
#[derive(Debug, Clone, Deserialize)]
pub struct ManifestEntry {
    pub arm_id: usize,
    pub selector: String,
    pub variant: String,
    pub jitcode_index: usize,
    pub debug_name: String,
}

/// Lazily deserialize the embedded manifest JSON on first access.
fn entries_storage() -> &'static Vec<ManifestEntry> {
    static CACHE: OnceLock<Vec<ManifestEntry>> = OnceLock::new();
    CACHE.get_or_init(|| {
        serde_json::from_str(OPCODE_DISPATCH_MANIFEST_JSON).unwrap_or_else(|e| {
            panic!("pyre_opcode_dispatch_manifest.json deserialize: {e}");
        })
    })
}

/// Reverse index: `Instruction::*` variant key → entry index.
///
/// Multi-pattern arms (e.g. `Instruction::LoadFast | Instruction::LoadFastBorrow`)
/// already produce one manifest entry per variant at build time, so this
/// index is a flat 1:1 map. Phase D's `MIFrame::dispatch_jitcode` will use
/// this for O(1) lookup of the assembled handler.
fn variant_to_entry_index() -> &'static HashMap<String, usize> {
    static CACHE: OnceLock<HashMap<String, usize>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let mut idx = HashMap::new();
        for (i, entry) in entries_storage().iter().enumerate() {
            idx.insert(entry.variant.clone(), i);
        }
        idx
    })
}

/// Number of variant entries in the manifest.
pub fn len() -> usize {
    entries_storage().len()
}

/// Iterate all manifest entries.
pub fn entries() -> &'static [ManifestEntry] {
    entries_storage()
}

/// Look up a manifest entry by `Instruction::*` variant key.
pub fn find_by_variant(variant_key: &str) -> Option<&'static ManifestEntry> {
    let i = *variant_to_entry_index().get(variant_key)?;
    Some(&entries_storage()[i])
}

/// Look up the assembled handler `jitcode_index` for an `Instruction::*`
/// variant. Returns `None` if the variant has no registered handler
/// (e.g. wildcard arms).
pub fn variant_to_jitcode_index(variant_key: &str) -> Option<usize> {
    find_by_variant(variant_key).map(|e| e.jitcode_index)
}

/// Validate the manifest at startup. Returns the total entry count on
/// success. Surfaces build-pipeline regressions before they hit the trace
/// path.
///
/// Checks:
/// 1. The manifest is non-empty.
/// 2. Every variant key is unique (multi-pattern arms must have been
///    expanded by `build.rs`).
/// 3. Every entry's `arm_id`, `jitcode_index`, and `variant` are non-empty.
pub fn validate() -> Result<usize, String> {
    let es = entries_storage();
    if es.is_empty() {
        return Err("pyre_opcode_dispatch_manifest.json is empty".to_string());
    }
    let mut seen = std::collections::HashSet::new();
    for (i, e) in es.iter().enumerate() {
        if e.variant.is_empty() {
            return Err(format!("entry {i} has empty variant key"));
        }
        if !seen.insert(&e.variant) {
            return Err(format!(
                "duplicate variant {:?} at entry {i} (manifest expansion bug)",
                e.variant
            ));
        }
    }
    let idx = variant_to_entry_index();
    if idx.len() != es.len() {
        return Err(format!(
            "variant index has {} entries but manifest has {}",
            idx.len(),
            es.len()
        ));
    }
    Ok(es.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_some_entries() {
        let n = len();
        assert!(
            n > 0,
            "expected at least one opcode dispatch manifest entry, got {n}"
        );
    }

    #[test]
    fn entries_have_jitcode_index_and_variant() {
        for entry in entries() {
            assert!(
                !entry.variant.is_empty(),
                "arm_id {} has empty variant",
                entry.arm_id
            );
        }
    }

    #[test]
    fn variant_index_round_trips() {
        for entry in entries() {
            let found = find_by_variant(&entry.variant)
                .unwrap_or_else(|| panic!("variant {:?} missing from index", entry.variant));
            assert_eq!(found.variant, entry.variant);
            assert_eq!(found.jitcode_index, entry.jitcode_index);
            assert_eq!(found.arm_id, entry.arm_id);
        }
    }

    #[test]
    fn validate_succeeds_at_startup() {
        let total = validate().expect("dispatch_manifest validation failed");
        assert!(total > 0, "validate() reported zero total entries");
    }

    #[test]
    fn multi_pattern_arms_share_jitcode_index() {
        // Multi-pattern arms (e.g. ExtendedArg | Resume | Nop | ...) must
        // map all their variants to the same jitcode_index by construction.
        let mut by_arm: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for entry in entries() {
            by_arm
                .entry(entry.arm_id)
                .or_default()
                .push(entry.jitcode_index);
        }
        for (arm_id, indices) in &by_arm {
            let unique: std::collections::HashSet<_> = indices.iter().copied().collect();
            assert_eq!(
                unique.len(),
                1,
                "arm_id {arm_id} variants point to multiple jitcode_indices: {indices:?}",
            );
        }
    }
}
