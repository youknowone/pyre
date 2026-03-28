//! SSA renamer for vectorization loop unrolling.
//!
//! Mirrors RPython's `optimizeopt/renamer.py`.
//! Used during loop unrolling to rename OpRefs from one iteration to the next.

use std::collections::HashMap;

use majit_ir::{Op, OpRef};

/// renamer.py:3-58: Renamer — maps old OpRefs to new OpRefs during unrolling.
pub struct Renamer {
    rename_map: HashMap<OpRef, OpRef>,
}

impl Renamer {
    pub fn new() -> Self {
        Renamer {
            rename_map: HashMap::new(),
        }
    }

    /// renamer.py:7-8: rename_box — look up the renamed OpRef.
    /// Returns the original if no mapping exists.
    pub fn rename_box(&self, opref: OpRef) -> OpRef {
        self.rename_map.get(&opref).copied().unwrap_or(opref)
    }

    /// renamer.py:10-18: start_renaming — register a mapping from var to tovar.
    pub fn start_renaming(&mut self, var: OpRef, tovar: OpRef) {
        // renamer.py:16-17: don't rename constants.
        if tovar.is_constant() {
            return;
        }
        self.rename_map.insert(var, tovar);
    }

    /// renamer.py:20-31: rename — apply renaming to all args and fail_args of an op.
    pub fn rename(&self, op: &mut Op) -> bool {
        for arg in op.args.iter_mut() {
            if let Some(&renamed) = self.rename_map.get(arg) {
                *arg = renamed;
            }
        }

        if op.opcode.is_guard() {
            // renamer.py:27: TODO op.rd_snapshot = self.rename_rd_snapshot(...)
            // renamer.py:28-29: failargs = self.rename_failargs(op, clone=True)
            if let Some(ref mut fail_args) = op.fail_args {
                let cloned: Vec<OpRef> = fail_args
                    .iter()
                    .map(|arg| self.rename_map.get(arg).copied().unwrap_or(*arg))
                    .collect();
                fail_args.clear();
                fail_args.extend(cloned);
            }
        }

        true
    }

    /// renamer.py:33-42: rename_failargs — rename a slice of fail_args.
    pub fn rename_failargs(&self, fail_args: &[OpRef]) -> Vec<OpRef> {
        fail_args
            .iter()
            .map(|arg| self.rename_map.get(arg).copied().unwrap_or(*arg))
            .collect()
    }

    /// renamer.py:44-57: rename_rd_snapshot — recursively rename snapshot boxes.
    /// In RPython, snapshots are nested MIFrame structures. In majit, resume data
    /// uses rd_numb (compact varint encoding), so this is a no-op for now,
    /// matching RPython's own TODO comment at renamer.py:27.
    pub fn rename_rd_snapshot(&self, _rd_numb: &Option<Vec<u8>>) -> Option<Vec<u8>> {
        // RPython: TODO op.rd_snapshot = self.rename_rd_snapshot(op.rd_snapshot, clone=True)
        // Not yet implemented in RPython's vector optimizer either.
        None
    }
}
