//! RPython `rpython/translator/translator.py` — `TranslationContext`
//! skeleton.
//!
//! Upstream `TranslationContext` is a large driver object carrying
//! configuration, the annotator, the flowgraph cache, entry-point
//! bookkeeping, and translation-phase state. Only the subset the
//! annotator port currently consumes is declared here; additional
//! fields land as the driver calls them.

use std::cell::RefCell;

use crate::flowspace::model::{BlockKey, GraphRef};

/// RPython `class TranslationContext` (translator.py:...).
///
/// Held as an `Option<TranslationContext>` by [`RPythonAnnotator`]
/// while the full port is in progress. Fields land as the annotator
/// driver's calls into `self.translator` manifest (`entry_point_graph`,
/// `annotator`, `graphs`, `update_call_graph`, ...).
pub struct TranslationContext {
    /// RPython `self.graphs = []` — every flow graph known to the
    /// translator. `RPythonAnnotator.complete()` iterates this to force
    /// annotation of each return variable.
    pub graphs: RefCell<Vec<GraphRef>>,
    /// RPython `self.entry_point_graph`. Set by
    /// `RPythonAnnotator.build_types(main_entry_point=True)`.
    pub entry_point_graph: RefCell<Option<GraphRef>>,
}

impl TranslationContext {
    pub fn new() -> Self {
        TranslationContext {
            graphs: RefCell::new(Vec::new()),
            entry_point_graph: RefCell::new(None),
        }
    }

    /// RPython `TranslationContext.update_call_graph(parent, child, tag)`
    /// (translator.py:...). Stub until the caller graph wiring is
    /// ported; the annotator's `recursivecall` invokes this when the
    /// `whence` tag is non-None.
    pub fn update_call_graph(
        &self,
        _parent: &GraphRef,
        _child: &GraphRef,
        _tag: (BlockKey, usize),
    ) {
        // Stub — upstream maintains a call-graph multi-map keyed on
        // (caller, callee) with a set of tag payloads per edge.
    }
}

impl Default for TranslationContext {
    fn default() -> Self {
        Self::new()
    }
}
