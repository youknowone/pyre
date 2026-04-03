//! Register allocation for flattened JitCode — scaffold.
//!
//! RPython equivalent: `rpython/jit/codewriter/regalloc.py`.
//!
//! **Status: scaffold.** Currently assigns one register per value (no reuse).
//! RPython's regalloc.py delegates to `rpython.tool.algo.regalloc` which does
//! actual graph-coloring with interference analysis and register reuse.
//! This module will need the same when JitCode bytecode encoding is implemented.

use std::collections::HashMap;

use crate::graph::ValueId;
use crate::passes::flatten::{FlatOp, FlattenedFunction, RegKind};

/// Result of register allocation for one kind.
///
/// RPython: `regalloc.perform_register_allocation()` returns an object
/// with a `_coloring` dict mapping variables to register indices.
#[derive(Debug, Clone)]
pub struct RegAllocResult {
    /// RPython: `RegAlloc._coloring` — maps ValueId → register index.
    pub coloring: HashMap<ValueId, usize>,
    /// Number of registers used for this kind.
    pub num_regs: usize,
}

/// Perform register allocation for a single kind on a flattened function.
///
/// RPython: `regalloc.perform_register_allocation(graph, kind)`.
///
/// This is a simple linear-scan approach: each value gets its own register.
/// A full interference-graph coloring would reduce register count but is
/// not needed for correctness.
pub fn perform_register_allocation(flattened: &FlattenedFunction, kind: RegKind) -> RegAllocResult {
    let mut coloring = HashMap::new();
    let mut next_reg = 0usize;

    for (vid, vkind) in &flattened.value_kinds {
        if *vkind == kind {
            coloring.insert(*vid, next_reg);
            next_reg += 1;
        }
    }

    RegAllocResult {
        coloring,
        num_regs: next_reg,
    }
}

/// Perform register allocation for all three kinds.
///
/// RPython: codewriter.py lines 45-47:
/// ```python
/// for kind in KINDS:
///     regallocs[kind] = perform_register_allocation(graph, kind)
/// ```
pub fn perform_all_register_allocations(
    flattened: &FlattenedFunction,
) -> HashMap<RegKind, RegAllocResult> {
    let mut result = HashMap::new();
    for kind in [RegKind::Int, RegKind::Ref, RegKind::Float] {
        result.insert(kind, perform_register_allocation(flattened, kind));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_separates_by_kind() {
        let mut flat = FlattenedFunction {
            name: "test".into(),
            ops: vec![],
            num_values: 3,
            num_blocks: 1,
            value_kinds: HashMap::new(),
        };
        flat.value_kinds.insert(ValueId(0), RegKind::Int);
        flat.value_kinds.insert(ValueId(1), RegKind::Ref);
        flat.value_kinds.insert(ValueId(2), RegKind::Int);

        let allocs = perform_all_register_allocations(&flat);
        assert_eq!(allocs[&RegKind::Int].num_regs, 2);
        assert_eq!(allocs[&RegKind::Ref].num_regs, 1);
        assert_eq!(allocs[&RegKind::Float].num_regs, 0);
    }
}
