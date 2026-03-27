//! SSA renamer for vectorization loop unrolling.
//!
//! Mirrors RPython's `optimizeopt/renamer.py`.
//! Used during loop unrolling to rename OpRefs from one iteration to the next.

use std::collections::HashMap;

use majit_ir::{Op, OpRef};

/// renamer.py:3-18: Renamer — maps old OpRefs to new OpRefs during unrolling.
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
        // renamer.py:16-17: don't rename constants (is_constant check).
        // In majit, constants have OpRef.0 >= 10_000.
        if tovar.0 >= 10_000 {
            return;
        }
        self.rename_map.insert(var, tovar);
    }

    /// renamer.py:20-31: rename — apply renaming to all args and fail_args of an op.
    pub fn rename(&self, op: &mut Op) {
        for arg in op.args.iter_mut() {
            if let Some(&renamed) = self.rename_map.get(arg) {
                *arg = renamed;
            }
        }

        // renamer.py:25-29: rename fail_args for guard ops.
        if op.opcode.is_guard() {
            if let Some(ref mut fail_args) = op.fail_args {
                for arg in fail_args.iter_mut() {
                    if let Some(&renamed) = self.rename_map.get(arg) {
                        *arg = renamed;
                    }
                }
            }
        }
    }

    /// renamer.py:33-42: rename_failargs — rename fail_args in place or cloned.
    pub fn rename_failargs(&self, fail_args: &[OpRef]) -> Vec<OpRef> {
        fail_args
            .iter()
            .map(|arg| self.rename_map.get(arg).copied().unwrap_or(*arg))
            .collect()
    }
}
