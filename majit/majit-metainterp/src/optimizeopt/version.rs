//! Loop versioning.
//!
//! Mirrors RPython's `version.py`: LoopVersionInfo and LoopVersion.
//! When a guard is known to fail frequently, a specialized version of
//! the loop can be compiled and stitched directly to that guard.

use std::collections::HashMap;

use majit_ir::Op;

/// Tracks multiple compiled versions of a loop, keyed by guard fail_index.
///
/// version.py: LoopVersionInfo(BasicLoopInfo)
pub struct LoopVersionInfo {
    /// Guard fail indices in insertion order.
    pub descrs: Vec<u32>,
    /// Maps guard fail_index → the version that should run on failure.
    pub leads_to: HashMap<u32, LoopVersion>,
    /// Insertion index for `track()`. -1 means append.
    insert_index: i32,
}

impl LoopVersionInfo {
    pub fn new() -> Self {
        LoopVersionInfo {
            descrs: Vec::new(),
            leads_to: HashMap::new(),
            insert_index: -1,
        }
    }

    /// Mark the current position for subsequent `track()` insertions.
    pub fn mark(&mut self) {
        self.insert_index = self.descrs.len() as i32;
    }

    /// Clear the insertion mark.
    pub fn clear(&mut self) {
        self.insert_index = -1;
    }

    /// Track a guard and associate it with a loop version.
    pub fn track(&mut self, fail_index: u32, version: LoopVersion) {
        let i = self.insert_index;
        if i >= 0 {
            self.descrs.insert(i as usize, fail_index);
        } else {
            self.descrs.push(fail_index);
        }
        self.leads_to.insert(fail_index, version);
    }

    /// Remove a guard from tracking.
    pub fn remove(&mut self, fail_index: u32) {
        self.leads_to.remove(&fail_index);
        self.descrs.retain(|d| *d != fail_index);
    }

    /// Look up the version for a guard.
    pub fn get(&self, fail_index: u32) -> Option<&LoopVersion> {
        self.leads_to.get(&fail_index)
    }

    /// Number of tracked versions.
    pub fn num_versions(&self) -> usize {
        self.leads_to.len()
    }
}

/// A specialized version of a loop body, attached to a guard.
///
/// version.py: LoopVersion
#[derive(Clone, Debug)]
pub struct LoopVersion {
    /// The specialized loop body operations.
    pub ops: Vec<Op>,
    /// Input arguments for this version.
    pub inputargs: Vec<majit_ir::InputArg>,
}

impl LoopVersion {
    pub fn new(ops: Vec<Op>, inputargs: Vec<majit_ir::InputArg>) -> Self {
        LoopVersion { ops, inputargs }
    }
}
