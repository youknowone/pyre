//! Port boundary for `rpython/translator/backendopt/malloc.py`.
//!
//! Upstream `all.py` imports `remove_mallocs` from this module and calls
//! it from `inline_malloc_removal_phase`. Pyre keeps the same module and
//! function boundary even while the 566-LOC escape-analysis pass itself is
//! still unported, so the missing pass is explicit and the default
//! `mallocs=True` pipeline keeps failing at the correct upstream point.

use std::rc::Rc;

use crate::flowspace::model::GraphRef;
use crate::translator::tool::taskengine::TaskError;
use crate::translator::translator::TranslationContext;

/// RPython `remove_mallocs(translator, graphs)` at `malloc.py:553-566`.
///
/// The real implementation wraps `LLTypeMallocRemover` (`malloc.py:333-547`),
/// which inherits `BaseMallocRemover` (`malloc.py:26-332`) and uses
/// `LifeTime` (`malloc.py:9-24`) to prove non-escaping allocations can be
/// replaced by direct field variables.
pub fn remove_mallocs(
    _translator: &Rc<TranslationContext>,
    _graphs: &[GraphRef],
) -> Result<(), TaskError> {
    Err(TaskError {
        message: "malloc.py:553 remove_mallocs: TODO — \
                  malloc.py (566 LOC LLTypeMallocRemover / \
                  BaseMallocRemover escape-analysis pass) is unported. \
                  Upstream default has mallocs=True so the default backendopt \
                  pipeline currently surfaces this gate. Convergence path: \
                  port malloc.py:9-566 verbatim here (UnionFind / simplify / \
                  removenoops / lltype.Struct deps already landed locally)."
            .to_string(),
    })
}
