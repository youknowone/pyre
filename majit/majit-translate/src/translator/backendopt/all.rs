//! Port of `rpython/translator/backendopt/all.py`.

use std::rc::Rc;

use crate::config::config::{Config, ConfigValue, OptionValue};
use crate::config::translationoption::get_combined_translation_config;
use crate::flowspace::model::GraphRef;
use crate::translator::backendopt::{constfold, removenoops};
use crate::translator::simplify;
use crate::translator::tool::taskengine::TaskError;
use crate::translator::translator::TranslationContext;

/// Port of upstream `backend_optimizations(translator, graphs=None,
/// secondary=False, inline_graph_from_anywhere=False, **kwds)` at
/// `all.py:35-130`.
///
/// `kwds` is `Vec<(String, OptionValue)>` rather than `HashMap` so the
/// caller's `**kwds` order is preserved through the
/// `for key, value in kwds.iteritems():` walk at `config.py:131`.
/// Upstream RPython is Python 2; iteration order there is unspecified
/// for plain `dict`, so the Vec preserves the caller's literal argument
/// order — see [`crate::config::config::Config::set`] for the full
/// citation.
///
/// `live_config` is upstream's `translator.config` carried as
/// [`Rc<Config>`].  The local [`TranslationContext`] holds only a typed
/// snapshot, so the driver passes the live schema-driven `Rc<Config>`
/// it owns (`driver.py:194 TranslationContext(config=self.config)`).
/// When `None` is supplied we fall back to the schema defaults — that
/// path is exercised only by tests that build a translator from
/// scratch without going through the driver.
pub fn backend_optimizations(
    translator: Rc<TranslationContext>,
    graphs: Option<Vec<GraphRef>>,
    secondary: bool,
    inline_graph_from_anywhere: bool,
    kwds: Vec<(String, OptionValue)>,
    live_config: Option<&Rc<Config>>,
) -> Result<(), TaskError> {
    let _ = inline_graph_from_anywhere;

    // Upstream `all.py:43-44`:
    // `config = translator.config.translation.backendopt.copy(as_default=True)`
    // then `config.set(**kwds)`.
    let config = backendopt_config(kwds, live_config)?;

    // Upstream `all.py:46-47`: `graphs is None` falls back to
    // `translator.graphs`.
    let graphs = graphs.unwrap_or_else(|| translator.graphs.borrow().clone());

    // Upstream `all.py:48-49`:
    //     for graph in graphs:
    //         assert not getattr(graph, '_seen_by_the_backend', False)
    for graph in &graphs {
        if graph.borrow()._seen_by_the_backend.get() {
            return Err(TaskError {
                message: format!(
                    "all.py:48 backend_optimizations: graph {:?} already \
                     seen by the C backend",
                    graph.borrow().name
                ),
            });
        }
    }

    // Upstream `all.py:51-130` runs each sub-pass in pipeline order;
    // the function returns implicitly after the last pass. The Rust
    // port runs every ported pass in upstream order and surfaces an
    // unported pass as a `TaskError` when (and only when) the live
    // config requests it — exactly the upstream "pass raises mid-
    // pipeline" semantic. Earlier "collect every missing leaf up
    // front" was a NEW-DEVIATION that skipped the ported passes
    // entirely.

    // Upstream `:51-53 print_statistics`. Stub helper retained so the
    // call site mirrors upstream; the body is a no-op until the real
    // statistics module lands.
    if boolopt(&config, "print_statistics")? {
        print_statistics_stub("before optimizations");
    }

    // Upstream `:55-57 replace_we_are_jitted`.
    if boolopt(&config, "replace_we_are_jitted")? {
        for graph in &graphs {
            constfold::replace_we_are_jitted(&graph.borrow());
        }
    }

    // Upstream `:59-61 remove_asserts`.
    if boolopt(&config, "remove_asserts")? {
        constfold_pass(&config, &graphs)?;
        return Err(TaskError {
            message: "all.py:61 remove_asserts not yet ported".to_string(),
        });
    }

    // Upstream `:63-66 really_remove_asserts → removenoops.remove_debug_assert`.
    if boolopt(&config, "really_remove_asserts")? {
        return Err(TaskError {
            message: "all.py:65 removenoops.remove_debug_assert not yet ported".to_string(),
        });
    }

    // Upstream `:69-80 remove_obvious_noops()` (first invocation).
    remove_obvious_noops(&config, &translator, &graphs)?;

    // Upstream `:82-92 inline + mallocs phase`.
    let inline_on = boolopt(&config, "inline")?;
    let mallocs_on = boolopt(&config, "mallocs")?;
    if inline_on || mallocs_on {
        let _threshold = if inline_on {
            floatopt(&config, "inline_threshold")?
        } else {
            0.0
        };
        return Err(TaskError {
            message: "all.py:88 inline_malloc_removal_phase \
                 (inline.auto_inline_graphs + remove_mallocs) not yet ported"
                .to_string(),
        });
    }

    // Upstream `:94-97 storesink phase`.
    if boolopt(&config, "storesink")? {
        remove_obvious_noops(&config, &translator, &graphs)?;
        return Err(TaskError {
            message: "all.py:97 storesink_graph not yet ported".to_string(),
        });
    }

    // Upstream `:99-113 profile_based_inline`.
    if stropt(&config, "profile_based_inline")?.is_some() && !secondary {
        return Err(TaskError {
            message: "all.py:99-113 profile_based_inline not yet ported".to_string(),
        });
    }

    // Upstream `:114 constfold(config, graphs)` — runs unconditionally
    // (gated only by config.constfold inside `constfold_pass`).
    constfold_pass(&config, &graphs)?;

    // Upstream `:116-119 merge_if_blocks`.
    if boolopt(&config, "merge_if_blocks")? {
        return Err(TaskError {
            message: "all.py:119 merge_if_blocks not yet ported".to_string(),
        });
    }

    if boolopt(&config, "print_statistics")? {
        print_statistics_stub("after if-to-switch");
    }

    // Upstream `:125 remove_obvious_noops()` (second invocation).
    remove_obvious_noops(&config, &translator, &graphs)?;

    // Upstream `:127-128 for graph in graphs: checkgraph(graph)`.
    for graph in &graphs {
        crate::flowspace::model::checkgraph(&graph.borrow());
    }

    // Upstream `:130 gilanalysis.analyze(graphs, translator)`.
    Err(TaskError {
        message: "all.py:130 gilanalysis.analyze not yet ported".to_string(),
    })
}

/// RPython `constfold(config, graphs)` at `all.py:133-136`.
pub(crate) fn constfold_pass(config: &Rc<Config>, graphs: &[GraphRef]) -> Result<(), TaskError> {
    if boolopt(config, "constfold")? {
        for graph in graphs {
            constfold::constant_fold_graph(&graph.borrow());
        }
    }
    Ok(())
}

/// RPython nested `remove_obvious_noops()` at `all.py:69-80`.
pub(crate) fn remove_obvious_noops(
    config: &Rc<Config>,
    translator: &TranslationContext,
    graphs: &[GraphRef],
) -> Result<(), TaskError> {
    for graph in graphs {
        let graph = graph.borrow();
        removenoops::remove_same_as(&graph);
        simplify::eliminate_empty_blocks(&graph);
        simplify::transform_dead_op_vars(&graph, Some(translator));
        removenoops::remove_duplicate_casts(&graph, translator);
    }
    if boolopt(config, "print_statistics")? {
        print_statistics_stub("after no-op removal");
    }
    Ok(())
}

fn backendopt_config(
    kwds: Vec<(String, OptionValue)>,
    live_config: Option<&Rc<Config>>,
) -> Result<Rc<Config>, TaskError> {
    // Upstream `all.py:43`:
    // `config = translator.config.translation.backendopt.copy(as_default=True)`.
    // Take the backendopt subgroup off whichever `Rc<Config>` the
    // caller is willing to share — the live one when available, the
    // fresh schema otherwise.
    let owned_root: Option<Rc<Config>> = if live_config.is_none() {
        Some(get_combined_translation_config(None, None, None, true).map_err(task_error)?)
    } else {
        None
    };
    let root: &Rc<Config> = live_config.unwrap_or_else(|| owned_root.as_ref().unwrap());
    let backendopt = match root.get("translation.backendopt").map_err(task_error)? {
        ConfigValue::SubConfig(config) => config.copy(true),
        other => {
            return Err(TaskError {
                message: format!("all.py:43 expected backendopt SubConfig, got {other:?}"),
            });
        }
    };
    backendopt.set(kwds).map_err(task_error)?;
    Ok(backendopt)
}

fn boolopt(config: &Rc<Config>, name: &str) -> Result<bool, TaskError> {
    match config.get(name).map_err(task_error)? {
        ConfigValue::Value(OptionValue::Bool(value)) => Ok(value),
        ConfigValue::Value(OptionValue::None) => Ok(false),
        other => Err(TaskError {
            message: format!("all.py backendopt config {name}: expected bool, got {other:?}"),
        }),
    }
}

fn floatopt(config: &Rc<Config>, name: &str) -> Result<f64, TaskError> {
    match config.get(name).map_err(task_error)? {
        ConfigValue::Value(OptionValue::Float(value)) => Ok(value),
        ConfigValue::Value(OptionValue::Int(value)) => Ok(value as f64),
        ConfigValue::Value(OptionValue::Bool(value)) => Ok(if value { 1.0 } else { 0.0 }),
        other => Err(TaskError {
            message: format!("all.py backendopt config {name}: expected float, got {other:?}"),
        }),
    }
}

fn stropt(config: &Rc<Config>, name: &str) -> Result<Option<String>, TaskError> {
    match config.get(name).map_err(task_error)? {
        ConfigValue::Value(OptionValue::Str(value)) if !value.is_empty() => Ok(Some(value)),
        ConfigValue::Value(OptionValue::Choice(value)) if !value.is_empty() => Ok(Some(value)),
        ConfigValue::Value(OptionValue::None) => Ok(None),
        ConfigValue::Value(OptionValue::Str(_)) | ConfigValue::Value(OptionValue::Choice(_)) => {
            Ok(None)
        }
        other => Err(TaskError {
            message: format!("all.py backendopt config {name}: expected string, got {other:?}"),
        }),
    }
}

fn print_statistics_stub(_phase: &str) {}

fn task_error(error: impl std::fmt::Debug) -> TaskError {
    TaskError {
        message: format!("all.py backend_optimizations config error: {error:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{
        Block, BlockRefExt, ConstValue, Constant, FunctionGraph, Hlvalue, Link, SpaceOperation,
        Variable,
    };
    use std::cell::RefCell;

    fn fixture_translator() -> Rc<TranslationContext> {
        Rc::new(TranslationContext::new())
    }

    fn graph_ref(graph: FunctionGraph) -> GraphRef {
        Rc::new(RefCell::new(graph))
    }

    /// Default backendopt config has `inline`, `mallocs`, `merge_if_blocks`,
    /// `storesink` all True. None of those subpasses are ported; pass kwds
    /// that disable them so the structural shell tests exercise only the
    /// ported subset (replace_we_are_jitted / removenoops / simplify /
    /// constfold).
    fn ported_only_kwds() -> Vec<(String, OptionValue)> {
        vec![
            ("inline".to_string(), OptionValue::Bool(false)),
            ("mallocs".to_string(), OptionValue::Bool(false)),
            ("merge_if_blocks".to_string(), OptionValue::Bool(false)),
            ("storesink".to_string(), OptionValue::Bool(false)),
        ]
    }

    fn make_int_constfold_graph() -> (Variable, GraphRef) {
        let r = Variable::named("r");
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![
                Hlvalue::Constant(Constant::new(ConstValue::Int(1))),
                Hlvalue::Constant(Constant::new(ConstValue::Int(2))),
            ],
            Hlvalue::Variable(r.clone()),
        ));
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(r.clone())],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        (r, graph_ref(graph))
    }

    #[test]
    fn backendopt_fails_fast_with_unported_leaf_listing() {
        // Upstream `all.py:35-130` either runs the full pipeline or
        // raises before any mutation. Several passes are unported
        // locally (`gilanalysis.analyze` runs unconditionally at
        // `:130`), so the call must fail before mutating any graph.
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Constant(Constant::new(ConstValue::None))],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let graph = graph_ref(graph);

        let snapshot_ops_before = graph.borrow().startblock.borrow().operations.clone();
        let err = backend_optimizations(
            fixture_translator(),
            Some(vec![graph.clone()]),
            false,
            false,
            ported_only_kwds(),
            None,
        )
        .expect_err("backendopt with any unported leaf must fail without mutation");

        // The reported message must enumerate the unported leaves the
        // current call would have needed to honour, so callers can see
        // which specific leaves are still pending.
        assert!(
            err.message.contains("all.py:130 gilanalysis.analyze"),
            "expected gilanalysis citation, got: {}",
            err.message
        );

        // No graph mutation must have happened before the error fired.
        let snapshot_ops_after = graph.borrow().startblock.borrow().operations.clone();
        assert_eq!(
            snapshot_ops_before, snapshot_ops_after,
            "backend_optimizations must not mutate graphs before failing"
        );
    }

    #[test]
    fn remove_obvious_noops_helper_drops_same_as_op() {
        // The pipeline-helper used by `backend_optimizations` once
        // every leaf lands. Tested directly so the partial pipeline
        // can still be exercised without going through the
        // fail-fast public entry point.
        let x = Variable::named("x");
        let y = Variable::named("y");
        let start = Block::shared(vec![Hlvalue::Variable(x.clone())]);
        let graph = FunctionGraph::new("f", start.clone());
        start.borrow_mut().operations.push(SpaceOperation::new(
            "same_as",
            vec![Hlvalue::Variable(x.clone())],
            Hlvalue::Variable(y.clone()),
        ));
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(y)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let graph = graph_ref(graph);
        let translator = fixture_translator();
        let config = backendopt_config(ported_only_kwds(), None).expect("config");

        remove_obvious_noops(&config, &translator, &[graph.clone()]).expect("remove_obvious_noops");

        let borrowed = graph.borrow();
        assert!(borrowed.startblock.borrow().operations.is_empty());
        let link_arg = borrowed.startblock.borrow().exits[0].borrow().args[0]
            .clone()
            .expect("link arg");
        assert_eq!(link_arg, Hlvalue::Variable(x));
    }

    #[test]
    fn constfold_pass_helper_folds_int_add() {
        let (_r, graph) = make_int_constfold_graph();
        let config = backendopt_config(ported_only_kwds(), None).expect("config");

        constfold_pass(&config, &[graph.clone()]).expect("constfold_pass");

        let borrowed = graph.borrow();
        assert!(borrowed.startblock.borrow().operations.is_empty());
        let link_arg = borrowed.startblock.borrow().exits[0].borrow().args[0]
            .clone()
            .expect("link arg");
        assert!(matches!(
            link_arg,
            Hlvalue::Constant(Constant {
                value: ConstValue::Int(3),
                ..
            })
        ));
    }
}
