//! RPython `rpython/translator/translator.py` — `TranslationContext`
//! skeleton.
//!
//! Upstream `TranslationContext` is a large driver object carrying
//! configuration, the annotator, the flowgraph cache, entry-point
//! bookkeeping, and translation-phase state. Only the subset the
//! annotator port currently consumes is declared here; additional
//! fields land as the driver calls them.

/// RPython `class TranslationContext` (translator.py:...).
///
/// Held as an `Option<TranslationContext>` by [`RPythonAnnotator`]
/// while the full port is in progress. Empty struct for now; fields
/// land as the annotator driver's calls into `self.translator`
/// manifest (`entry_point_graph`, `annotator`, `graphs`,
/// `update_call_graph`, ...).
pub struct TranslationContext;

impl TranslationContext {
    pub fn new() -> Self {
        TranslationContext
    }
}

impl Default for TranslationContext {
    fn default() -> Self {
        Self::new()
    }
}
