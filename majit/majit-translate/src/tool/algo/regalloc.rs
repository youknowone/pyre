//! Re-export of `rpython/tool/algo/regalloc.py`.
//!
//! PyPy's `jit/codewriter/regalloc.py` is a tiny wrapper around this module.
//! The Rust implementation historically lives in `jit_codewriter::regalloc`
//! because its concrete API is keyed by codewriter [`RegKind`].  Keep this
//! module path so callers can use the same source layout as RPython.

pub use crate::jit_codewriter::regalloc::{
    RegAllocResult, augment_canonical_exceptblock_on_graph, perform_all_register_allocations,
    perform_register_allocation,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flatten::RegKind;
    use crate::model::{ConcreteType, FunctionGraph, OpKind, ValueType};

    #[test]
    fn perform_register_allocation_matches_codewriter_entrypoint() {
        let mut graph = FunctionGraph::new("tool_algo_regalloc_test");
        let entry = graph.startblock;
        let v0 = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "a".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let v1 = graph
            .push_op_var(
                entry,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs: v0.clone(),
                    rhs: v0.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(v1.clone()));

        FunctionGraph::set_concretetype_of_inline(&v0, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&v1, ConcreteType::Signed);

        let result = perform_register_allocation(&graph, RegKind::Int);
        assert_eq!(result.num_regs, 1);
        assert_eq!(
            result.color_for_variable(&v0),
            result.color_for_variable(&v1)
        );
    }
}
