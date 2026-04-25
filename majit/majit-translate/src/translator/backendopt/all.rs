//! Port of `rpython/translator/backendopt/all.py`.
//!
//! Upstream is 164 LOC of one entry point — `backend_optimizations` —
//! plus a handful of helpers (`constfold`, `inline_malloc_removal_phase`,
//! `print_statistics`, `cleanup_graphs`). The driver task
//! `task_backendopt_lltype` (`driver.py:380-384`) calls only
//! `backend_optimizations(translator, replace_we_are_jitted=True)`.
//!
//! The local port keeps the call shape and config-key handling
//! line-by-line with upstream `:35-50`; the body returns
//! [`TaskError`] until the underlying passes
//! (`replace_we_are_jitted`, `constfold`, `remove_asserts`,
//! `removenoops.*`, `simplify.*`, `inline_malloc_removal_phase`,
//! `storesink_graph`, `merge_if_blocks`, `gilanalysis.analyze`,
//! `checkgraph`) land.

use std::collections::HashMap;
use std::rc::Rc;

use crate::config::config::OptionValue;
use crate::translator::tool::taskengine::TaskError;
use crate::translator::translator::TranslationContext;

/// Port of upstream `backend_optimizations(translator, graphs=None,
/// secondary=False, inline_graph_from_anywhere=False, **kwds)` at
/// `:35-130`.
///
/// The local entry mirrors upstream's argument shape — `graphs=None` ↔
/// `Option<Vec<...>>::None` falls back to `translator.graphs`,
/// `secondary` ↔ `bool`, `inline_graph_from_anywhere` ↔ `bool`, `kwds`
/// ↔ a `HashMap<String, OptionValue>` of overrides applied to the
/// per-translation `backendopt` Config sub-tree.
pub fn backend_optimizations(
    translator: Rc<TranslationContext>,
    _graphs: Option<Vec<Rc<dyn std::any::Any>>>,
    _secondary: bool,
    _inline_graph_from_anywhere: bool,
    kwds: HashMap<String, OptionValue>,
) -> Result<(), TaskError> {
    // Upstream `:43-44`: `config = translator.config.translation
    // .backendopt.copy(as_default=True); config.set(**kwds)`. The local
    // port doesn't yet propagate the `as_default=True` semantics
    // (Config.copy is not ported), so we just acknowledge the kwds
    // request and surface a TaskError at the leaf body.
    let _ = (translator, kwds);
    Err(TaskError {
        message: "all.py:35 backend_optimizations — leaf passes (replace_we_are_jitted / constfold / remove_asserts / removenoops / simplify / inline_malloc_removal_phase / storesink_graph / merge_if_blocks / gilanalysis.analyze / checkgraph) not yet ported".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::translationoption::get_combined_translation_config;
    use crate::translator::translator::TranslationContext;

    fn fixture_translator() -> Rc<TranslationContext> {
        let _config = get_combined_translation_config(None, None, None, true).expect("config");
        Rc::new(TranslationContext::new())
    }

    #[test]
    fn backend_optimizations_returns_task_error_until_leaves_land() {
        // Upstream `:35 def backend_optimizations` — every body branch
        // depends on a yet-unported pass. The Rust shell surfaces a
        // TaskError citing `all.py:35` so the driver task can record
        // the gap without panicking.
        let mut kwds = HashMap::new();
        kwds.insert("replace_we_are_jitted".to_string(), OptionValue::Bool(true));
        let err = backend_optimizations(fixture_translator(), None, false, false, kwds)
            .expect_err("must be DEFERRED");
        assert!(err.message.contains("all.py:35"), "{}", err.message);
    }
}
