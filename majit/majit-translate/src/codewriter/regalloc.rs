//! Codewriter register-allocation entry point.
//!
//! PyPy's `rpython/jit/codewriter/regalloc.py` is a tiny wrapper around
//! `rpython/tool/algo/regalloc.py`, adding only the codewriter-specific
//! `kind` predicate (`history.getkind(v.concretetype) == kind`) and the
//! `ListOfKind` carrier from `flatten.py`.
//!
//! The allocator implementation therefore lives in
//! [`crate::tool::algo::regalloc`]. This module keeps the upstream
//! codewriter path and public function name.

use crate::flatten::RegKind;
use crate::model::FunctionGraph;

pub use crate::tool::algo::regalloc::RegAllocator;

/// `regalloc.py::perform_register_allocation(graph, kind)` wrapper.
pub fn perform_register_allocation(graph: &FunctionGraph, kind: RegKind) -> RegAllocator {
    crate::tool::algo::regalloc::perform_register_allocation(graph, kind)
}

#[allow(dead_code)]
pub(crate) fn augment_canonical_exceptblock_on_graph(graph: &mut FunctionGraph) {
    crate::tool::algo::regalloc::augment_canonical_exceptblock_on_graph(graph)
}

#[allow(dead_code)]
pub(crate) fn perform_all_register_allocations(
    graph: &FunctionGraph,
) -> std::collections::HashMap<RegKind, RegAllocator> {
    crate::tool::algo::regalloc::perform_all_register_allocations(graph)
}
