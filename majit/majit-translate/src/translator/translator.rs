//! RPython `rpython/translator/translator.py` — `TranslationContext`
//! skeleton.
//!
//! Upstream `TranslationContext` is a large driver object carrying
//! configuration, the annotator, the flowgraph cache, entry-point
//! bookkeeping, and translation-phase state. Only the subset the
//! annotator port currently consumes is declared here; additional
//! fields land as the driver calls them.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::flowspace::model::{BlockKey, GraphKey, GraphRef};

/// Key for [`TranslationContext::callgraph`]. Matches upstream's
/// Python dict key `(caller_graph, callee_graph, position_tag)` at
/// translator.py:66, where `position_tag = (parent_block, parent_index)`.
/// All three components carry pointer-identity semantics; Rust uses
/// `GraphKey` / `BlockKey` for the object handles.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CallGraphKey {
    pub caller: GraphKey,
    pub callee: GraphKey,
    pub tag_block: BlockKey,
    pub tag_index: usize,
}

/// Value for [`TranslationContext::callgraph`]. Upstream stores
/// `(caller_graph, callee_graph)` (translator.py:67), preserving the
/// GraphRef handles that outlive individual call-site traversals.
#[derive(Clone)]
pub struct CallGraphEdge {
    pub caller: GraphRef,
    pub callee: GraphRef,
}

/// RPython `class TranslationContext` (translator.py:21-43).
///
/// Held by [`RPythonAnnotator`]; fields land incrementally as the
/// annotator driver's `self.translator.*` calls manifest.
pub struct TranslationContext {
    /// RPython `self.graphs = []` — every flow graph known to the
    /// translator. `RPythonAnnotator.complete()` iterates this to force
    /// annotation of each return variable.
    pub graphs: RefCell<Vec<GraphRef>>,
    /// RPython `self.entry_point_graph`. Set by
    /// `RPythonAnnotator.build_types(main_entry_point=True)`.
    pub entry_point_graph: RefCell<Option<GraphRef>>,
    /// RPython `self.callgraph = {}` (translator.py:41).
    /// `{opaque_tag: (caller-graph, callee-graph)}` — keyed by
    /// `(caller, callee, tag)` triple (translator.py:66). Populated
    /// every time the annotator's `recursivecall` records a non-None
    /// `whence` tag.
    pub callgraph: RefCell<HashMap<CallGraphKey, CallGraphEdge>>,
}

impl TranslationContext {
    pub fn new() -> Self {
        TranslationContext {
            graphs: RefCell::new(Vec::new()),
            entry_point_graph: RefCell::new(None),
            callgraph: RefCell::new(HashMap::new()),
        }
    }

    /// RPython `TranslationContext.update_call_graph(caller_graph,
    /// callee_graph, position_tag)` (translator.py:64-67).
    ///
    /// ```python
    /// def update_call_graph(self, caller_graph, callee_graph, position_tag):
    ///     key = caller_graph, callee_graph, position_tag
    ///     self.callgraph[key] = caller_graph, callee_graph
    /// ```
    ///
    /// Upstream dedupes by the full key triple; re-recording the same
    /// (caller, callee, tag) overwrites the value (same graph refs).
    pub fn update_call_graph(&self, caller: &GraphRef, callee: &GraphRef, tag: (BlockKey, usize)) {
        let (tag_block, tag_index) = tag;
        let key = CallGraphKey {
            caller: GraphKey::of(caller),
            callee: GraphKey::of(callee),
            tag_block,
            tag_index,
        };
        let edge = CallGraphEdge {
            caller: caller.clone(),
            callee: callee.clone(),
        };
        self.callgraph.borrow_mut().insert(key, edge);
    }
}

impl Default for TranslationContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{Block, FunctionGraph};
    use std::cell::RefCell as StdRefCell;
    use std::rc::Rc;

    fn mk_graph(name: &'static str) -> GraphRef {
        let block = Rc::new(StdRefCell::new(Block::new(vec![])));
        Rc::new(StdRefCell::new(FunctionGraph::new(name, block)))
    }

    #[test]
    fn update_call_graph_records_caller_callee_tag_triple() {
        let ctx = TranslationContext::new();
        let caller = mk_graph("caller");
        let callee = mk_graph("callee");
        let block = caller.borrow().startblock.clone();
        let tag = (BlockKey::of(&block), 7);
        ctx.update_call_graph(&caller, &callee, tag);
        assert_eq!(ctx.callgraph.borrow().len(), 1);
        let key = CallGraphKey {
            caller: GraphKey::of(&caller),
            callee: GraphKey::of(&callee),
            tag_block: BlockKey::of(&block),
            tag_index: 7,
        };
        assert!(ctx.callgraph.borrow().contains_key(&key));
    }

    #[test]
    fn update_call_graph_dedupes_on_full_triple() {
        let ctx = TranslationContext::new();
        let caller = mk_graph("caller");
        let callee = mk_graph("callee");
        let block = caller.borrow().startblock.clone();
        let tag = (BlockKey::of(&block), 0);
        ctx.update_call_graph(&caller, &callee, tag.clone());
        ctx.update_call_graph(&caller, &callee, tag);
        assert_eq!(ctx.callgraph.borrow().len(), 1);
    }

    #[test]
    fn update_call_graph_distinct_tags_record_distinct_edges() {
        let ctx = TranslationContext::new();
        let caller = mk_graph("caller");
        let callee = mk_graph("callee");
        let block = caller.borrow().startblock.clone();
        ctx.update_call_graph(&caller, &callee, (BlockKey::of(&block), 1));
        ctx.update_call_graph(&caller, &callee, (BlockKey::of(&block), 2));
        assert_eq!(ctx.callgraph.borrow().len(), 2);
    }
}
