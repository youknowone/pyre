//! Loop versioning.
//!
//! Mirrors RPython's `version.py`: LoopVersionInfo and LoopVersion.
//! When a guard is known to fail frequently, a specialized version of
//! the loop can be compiled and stitched directly to that guard.

use std::collections::HashMap;

use majit_ir::{DescrRef, Op, OpRef};

use crate::fail_descr;

/// version.py:9-86: LoopVersionInfo(BasicLoopInfo)
///
/// Tracks multiple compiled versions of a loop, keyed by guard descr.
/// `descrs` is the ordered list of guard descriptors that have loop versions.
/// `leads_to` maps each descr's fail_index to the LoopVersion.
pub struct LoopVersionInfo {
    /// version.py:19 — ordered list of fail indices for tracked guards.
    pub descrs: Vec<u32>,
    /// version.py:20 — maps fail_index → LoopVersion.
    pub leads_to: HashMap<u32, LoopVersion>,
    /// version.py:21 — insertion index for track(). -1 means append.
    insert_index: i32,
    /// version.py:22 — compiled loop versions.
    pub versions: Vec<LoopVersion>,
}

impl LoopVersionInfo {
    /// version.py:10-22
    pub fn new() -> Self {
        LoopVersionInfo {
            descrs: Vec::new(),
            leads_to: HashMap::new(),
            insert_index: -1,
            versions: Vec::new(),
        }
    }

    /// version.py:24-25: mark()
    pub fn mark(&mut self) {
        self.insert_index = self.descrs.len() as i32;
    }

    /// version.py:27-28: clear()
    pub fn clear(&mut self) {
        self.insert_index = -1;
    }

    /// version.py:30-36: track(op, descr, version)
    ///
    /// `fail_index` is the descr's unique identifier.
    pub fn track(&mut self, fail_index: u32, version: LoopVersion) {
        let i = self.insert_index;
        if i >= 0 {
            self.descrs.insert(i as usize, fail_index);
        } else {
            self.descrs.push(fail_index);
        }
        assert!(
            !self.leads_to.contains_key(&fail_index),
            "version.py:35 assert descr not in self.leads_to"
        );
        self.leads_to.insert(fail_index, version);
    }

    /// version.py:38-42: remove(descr)
    pub fn remove(&mut self, fail_index: u32) {
        if self.leads_to.remove(&fail_index).is_none() {
            panic!("version.py:42 could not remove fail_index={}", fail_index);
        }
        self.descrs.retain(|d| *d != fail_index);
    }

    /// version.py:44-45: get(descr)
    pub fn get(&self, fail_index: u32) -> Option<&LoopVersion> {
        self.leads_to.get(&fail_index)
    }

    /// version.py:47-53: snapshot(loop)
    ///
    /// Clone the loop operations and create a new LoopVersion.
    /// The version is registered in `self.versions`.
    pub fn snapshot(&mut self, ops: &[Op], label_args: &[OpRef]) -> LoopVersion {
        let newloop = ops.to_vec();
        let mut version = LoopVersion::new(newloop, label_args.to_vec());
        version.setup_once(self);
        self.versions.push(version.clone());
        version
    }

    /// version.py:55-86: post_loop_compilation(...)
    ///
    /// Compile each version and stitch to guard descriptors.
    /// This is called after the root trace is already compiled.
    ///
    /// In majit, the actual compilation is deferred to the backend.
    /// This method returns the list of (fail_index, LoopVersion) pairs
    /// that need to be compiled as bridges.
    pub fn pending_compilations(&self) -> Vec<(u32, LoopVersion)> {
        if self.versions.is_empty() {
            return Vec::new();
        }
        let mut result = Vec::new();
        for &fail_index in &self.descrs {
            if let Some(version) = self.leads_to.get(&fail_index) {
                result.push((fail_index, version.clone()));
            }
        }
        result
    }

    /// Number of tracked versions.
    pub fn num_versions(&self) -> usize {
        self.leads_to.len()
    }
}

impl Default for LoopVersionInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// version.py:89-122: LoopVersion
///
/// A specialized version of a trace loop, attached to a guard descriptor.
#[derive(Clone, Debug)]
pub struct LoopVersion {
    /// The specialized loop body operations.
    pub ops: Vec<Op>,
    /// Input arguments for this version (label arglist copy).
    pub inputargs: Vec<OpRef>,
}

impl LoopVersion {
    pub fn new(ops: Vec<Op>, inputargs: Vec<OpRef>) -> Self {
        LoopVersion { ops, inputargs }
    }

    /// version.py:99-112: setup_once(info)
    ///
    /// Clone guard descriptors in the version's ops so each version
    /// has its own descr identity. Re-track cloned loop-version guards.
    pub fn setup_once(&mut self, info: &mut LoopVersionInfo) {
        for op in &mut self.ops {
            if !op.opcode.is_guard() {
                continue;
            }
            let old_descr = match &op.descr {
                Some(d) => d.clone(),
                None => continue,
            };
            let Some(old_fd) = old_descr.as_fail_descr() else {
                continue;
            };
            let old_fail_index = old_fd.fail_index();
            // version.py:105: descr = olddescr.clone()
            let new_descr = fail_descr::make_resume_guard_descr(
                op.fail_args.as_ref().map_or(0, |fa| fa.len()),
                crate::resume::ResumeData {
                    vable_array: Vec::new(),
                    frames: Vec::new(),
                    virtuals: Vec::new(),
                    pending_fields: Vec::new(),
                },
            );
            op.descr = Some(new_descr.clone());
            // version.py:107-112: if descr.loop_version(), re-track
            // We check if the old descr was tracked in leads_to.
            if let Some(to_version) = info.leads_to.get(&old_fail_index).cloned() {
                let new_fail_index = new_descr
                    .as_fail_descr()
                    .expect("fresh descr must be FailDescr")
                    .fail_index();
                info.track(new_fail_index, to_version);
            }
        }
    }
}
