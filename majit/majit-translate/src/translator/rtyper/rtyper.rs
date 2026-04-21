//! RPython `rpython/rtyper/rtyper.py` — `RPythonTyper` skeleton.
//!
//! Only the surface currently consumed by `translator/simplify.py`
//! lands here: `translator.rtyper is None` and
//! `translator.rtyper.already_seen`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};

use crate::annotator::annrpython::RPythonAnnotator;
use crate::flowspace::model::{BlockKey, BlockRef};

/// RPython `class RPythonTyper(object)` (rtyper.py:42+).
///
/// The full constructor state lands incrementally as the rtyper port
/// progresses. For `simplify.py` parity we need the annotator link and
/// the `already_seen` dict populated by `specialize_more_blocks()`.
pub struct RPythonTyper {
    /// RPython `self.annotator`.
    ///
    /// Rust uses `Weak` to avoid an Rc cycle with
    /// `RPythonAnnotator.translator -> TranslationContext.rtyper`.
    pub annotator: Weak<RPythonAnnotator>,
    /// RPython `self.already_seen = {}` assigned in `specialize()`
    /// (rtyper.py:186). Membership is queried by `simplify.py`.
    pub already_seen: RefCell<HashMap<BlockKey, bool>>,
}

impl RPythonTyper {
    /// RPython `RPythonTyper.__init__(self, annotator, backend=...)`.
    ///
    /// Only the fields required by current simplify parity are seeded
    /// here; additional constructor state lands with the full rtyper
    /// port.
    pub fn new(annotator: &Rc<RPythonAnnotator>) -> Self {
        RPythonTyper {
            annotator: Rc::downgrade(annotator),
            already_seen: RefCell::new(HashMap::new()),
        }
    }

    /// Test/debug helper mirroring `self.already_seen[block] = True`
    /// in `specialize_more_blocks()` (rtyper.py:225).
    pub fn mark_already_seen(&self, block: &BlockRef) {
        self.already_seen
            .borrow_mut()
            .insert(BlockKey::of(block), true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::flowspace::model::BlockKey;

    #[test]
    fn new_rtyper_starts_with_empty_already_seen() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        assert!(rtyper.already_seen.borrow().is_empty());
    }

    #[test]
    fn mark_already_seen_records_block_key() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let block = ann.translator.borrow().entry_point_graph.borrow().clone();
        assert!(block.is_none());

        let graph = crate::flowspace::model::FunctionGraph::new(
            "g",
            std::rc::Rc::new(std::cell::RefCell::new(
                crate::flowspace::model::Block::new(vec![]),
            )),
        );
        let startblock = graph.startblock.clone();
        rtyper.mark_already_seen(&startblock);
        assert!(
            rtyper
                .already_seen
                .borrow()
                .contains_key(&BlockKey::of(&startblock))
        );
    }
}
