//! The graph-postprocessing slice of
//! `rpython/memory/gctransform/shadowstack.py`.
//!
//! `ShadowStackRootWalker.postprocess_graph` is the owner of
//! [`shadowcolor`].  Application functions are not decorated: the GC
//! transformer invokes [`ShadowStackFrameworkGCTransformer::inline_helpers_and_postprocess`]
//! once with its complete graph set, and this module applies the whole
//! `shadowcolor.py` pipeline to every graph.

use crate::flowspace::model::{GraphRef, Hlvalue};

use super::shadowcolor;

/// `shadowstack.py::ShadowStackRootWalker`'s postprocessing state.
///
/// The full root walker also owns stack walking and thread/stacklet support.
/// Those runtime pieces live in `majit-gc`; this translator-side carrier owns
/// only the `c_gcdata` constant consumed by `gc_enter_roots_frame`.
#[derive(Clone)]
pub struct ShadowStackRootWalker {
    c_const_gcdata: Hlvalue,
}

impl ShadowStackRootWalker {
    pub fn new(c_const_gcdata: Hlvalue) -> Self {
        Self { c_const_gcdata }
    }

    /// Strict port of `ShadowStackRootWalker.postprocess_graph`.
    pub fn postprocess_graph(
        &self,
        graph: &GraphRef,
        any_inlining: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut graph = graph.borrow_mut();
        if any_inlining {
            shadowcolor::postprocess_inlining(&graph)?;
        }
        let _use_push_pop =
            shadowcolor::postprocess_graph(&mut graph, self.c_const_gcdata.clone())?;
        // Upstream removes `graph` from `gct.graphs_to_inline` here when the
        // result is true.  The helper-inlining inventory is still empty in
        // this backend, so there is no entry to remove yet.
        Ok(())
    }
}

/// Graph-lifecycle owner corresponding to
/// `ShadowStackFrameworkGCTransformer` plus
/// `BaseGCTransformer.inline_helpers_and_postprocess`.
#[derive(Clone)]
#[expect(
    clippy::upper_case_acronyms,
    reason = "strict port of the upstream ShadowStackFrameworkGCTransformer symbol"
)]
pub struct ShadowStackFrameworkGCTransformer {
    root_walker: ShadowStackRootWalker,
}

impl ShadowStackFrameworkGCTransformer {
    pub fn new(c_const_gcdata: Hlvalue) -> Self {
        Self {
            root_walker: ShadowStackRootWalker::new(c_const_gcdata),
        }
    }

    /// `BaseGCTransformer.inline_helpers_and_postprocess` followed by
    /// `ShadowStackRootWalker.postprocess_graph`, over the complete graph set.
    ///
    /// `inline_helpers_into` has not yet materialised GC helper graphs in this
    /// backend, so `any_inlining` is false today.  Keeping it at the per-graph
    /// call site preserves the exact upstream ordering and gives the
    /// helper-inliner port one place to connect later.
    pub fn inline_helpers_and_postprocess(
        &self,
        graphs: &[GraphRef],
    ) -> Result<(), Box<dyn std::error::Error>> {
        for graph in graphs {
            let any_inlining = false;
            self.root_walker.postprocess_graph(graph, any_inlining)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use crate::flowspace::model::{
        Block, BlockRefExt, ConstValue, Constant, FunctionGraph, Hlvalue, SpaceOperation, Variable,
    };
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;

    use super::*;

    fn void_result() -> Hlvalue {
        Hlvalue::Variable(Variable::named("void"))
    }

    fn marker(name: &str, args: Vec<Hlvalue>) -> SpaceOperation {
        SpaceOperation::new(name, args, void_result())
    }

    fn graph(name: &str, with_roots: bool) -> GraphRef {
        let root_var = Variable::named("root");
        root_var.set_concretetype(Some(LowLevelType::Signed));
        let root = Hlvalue::Variable(root_var);
        let start = Block::shared(vec![root.clone()]);
        start.borrow_mut().operations = if with_roots {
            vec![
                marker("gc_push_roots", vec![root.clone()]),
                marker(
                    "direct_call",
                    vec![Hlvalue::Constant(Constant::new(ConstValue::Int(0)))],
                ),
                marker("gc_pop_roots", vec![root.clone()]),
            ]
        } else {
            vec![marker("int_add", vec![])]
        };
        let graph = FunctionGraph::new(name, start.clone());
        start.closeblock(vec![
            crate::flowspace::model::Link::new(vec![root], Some(graph.returnblock.clone()), None)
                .into_ref(),
        ]);
        Rc::new(RefCell::new(graph))
    }

    #[test]
    fn graph_set_is_postprocessed_without_function_attributes() {
        let first = graph("first", true);
        let second = graph("second", false);
        let c_gcdata = Hlvalue::Constant(Constant::new(ConstValue::Int(0)));
        let transformer = ShadowStackFrameworkGCTransformer::new(c_gcdata);

        transformer
            .inline_helpers_and_postprocess(&[first.clone(), second.clone()])
            .unwrap();
        let first_ops: Vec<String> = first
            .borrow()
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert!(first_ops.contains(&"gc_enter_roots_frame".to_string()));
        assert!(first_ops.contains(&"gc_save_root".to_string()));
        assert!(first_ops.contains(&"gc_restore_root".to_string()));
        assert!(!first_ops.contains(&"gc_push_roots".to_string()));
        assert!(!first_ops.contains(&"gc_pop_roots".to_string()));
        assert_eq!(
            second.borrow().startblock.borrow().operations[0].opname,
            "int_add"
        );
    }
}
