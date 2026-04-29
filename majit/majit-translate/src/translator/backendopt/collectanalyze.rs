//! Port of `rpython/translator/backendopt/collectanalyze.py`.
//!
//! `CollectAnalyzer` is the [`super::graphanalyze::BoolGraphAnalyzer`]
//! subclass that decides whether a call site can collect (i.e. trigger
//! a GC). Upstream consumers live in `rpython/memory/gctransform/`,
//! which pyre has not yet ported. This module is published as a parity
//! sibling alongside [`super::canraise`] and [`super::gilanalysis`] so
//! the surface lines up the day a consumer needs it.
//!
//! Per upstream:
//!
//! * `analyze_direct_call(self, graph, seen=None)` short-circuits to
//!   `bottom_result()` when the callee carries
//!   `_gctransformer_hint_cannot_collect_`, and to `top_result()`
//!   when it carries `_gctransformer_hint_close_stack_`. Otherwise
//!   falls through to the framework walk
//!   (`graphanalyze.BoolGraphAnalyzer.analyze_direct_call`).
//! * `analyze_external_call(self, funcobj, seen=None)` returns
//!   `top_result()` when `funcobj.random_effects_on_gcobjs` is set;
//!   otherwise delegates to the inherited handler — which in
//!   `BoolGraphAnalyzer` returns `top_result()` (same conservative
//!   default as `RaiseAnalyzer`'s `getattr(fnobj, 'canraise', True)`).
//! * `analyze_simple_operation(self, op, graphinfo)` returns `True`
//!   for `malloc` / `malloc_varsize` with `flavor='gc'`, and
//!   otherwise reads `LL_OPERATIONS[opname].canmallocgc`.

use crate::flowspace::model::{ConstValue, FunctionGraph, GraphRef, Hlvalue, SpaceOperation};
use crate::tool::algo::unionfind::UnionFind;
use crate::translator::backendopt::graphanalyze::{
    Dependency, DependencyTracker, GraphAnalyzer, framework_analyze_direct_call,
};
use crate::translator::rtyper::lltypesystem::lloperation::ll_operations;
use crate::translator::rtyper::lltypesystem::lltype::_ptr_obj;
use crate::translator::translator::TranslationContext;

/// `class CollectAnalyzer(graphanalyze.BoolGraphAnalyzer)` at
/// `collectanalyze.py:7-33`.
pub struct CollectAnalyzer<'t> {
    translator: &'t TranslationContext,
    /// Upstream `GraphAnalyzer._analyzed_calls` (`graphanalyze.py:13`).
    analyzed_calls: UnionFind<usize, Dependency<bool>>,
}

impl<'t> CollectAnalyzer<'t> {
    pub fn new(translator: &'t TranslationContext) -> Self {
        Self {
            translator,
            analyzed_calls: UnionFind::new(|_| Dependency::new(false)),
        }
    }
}

/// Helper for the `func._gctransformer_hint_*` short-circuits at
/// upstream `:11-18`. Returns `Some(verdict)` when one of the two
/// hints is set and the framework should bypass the call-graph walk;
/// `None` when neither hint is present.
fn graph_hint_verdict(graph: &FunctionGraph) -> Option<bool> {
    let func = graph.func.as_ref()?;
    if func._gctransformer_hint_cannot_collect_ {
        return Some(false);
    }
    if func._gctransformer_hint_close_stack_ {
        return Some(true);
    }
    None
}

impl<'t> GraphAnalyzer<bool, ()> for CollectAnalyzer<'t> {
    fn translator(&self) -> &TranslationContext {
        self.translator
    }

    fn analyzed_calls(&mut self) -> &mut UnionFind<usize, Dependency<bool>> {
        &mut self.analyzed_calls
    }

    /// Upstream `:9-20`:
    ///
    /// ```python
    /// def analyze_direct_call(self, graph, seen=None):
    ///     try:
    ///         func = graph.func
    ///     except AttributeError:
    ///         pass
    ///     else:
    ///         if getattr(func, '_gctransformer_hint_cannot_collect_', False):
    ///             return False
    ///         if getattr(func, '_gctransformer_hint_close_stack_', False):
    ///             return True
    ///     return graphanalyze.BoolGraphAnalyzer.analyze_direct_call(self,
    ///                                                               graph, seen)
    /// ```
    fn analyze_direct_call(
        &mut self,
        graph: &GraphRef,
        seen: Option<&mut DependencyTracker<bool>>,
    ) -> bool {
        if let Some(verdict) = graph_hint_verdict(&graph.borrow()) {
            return verdict;
        }
        framework_analyze_direct_call(self, graph, seen)
    }

    /// Upstream `:21-25`:
    ///
    /// ```python
    /// def analyze_external_call(self, funcobj, seen=None):
    ///     if funcobj.random_effects_on_gcobjs:
    ///         return True
    ///     return graphanalyze.BoolGraphAnalyzer.analyze_external_call(
    ///         self, funcobj, seen)
    /// ```
    ///
    /// `funcobj` upstream is `op.args[0].value._obj`; pyre routes
    /// through the same LLPtr unwrap that
    /// [`super::canraise::RaiseAnalyzer::analyze_external_call`] does.
    /// `random_effects_on_gcobjs` is read off `_func.attrs` — the
    /// same attribute mirror that carries `canraise`.
    fn analyze_external_call(
        &mut self,
        op: &SpaceOperation,
        _seen: Option<&mut DependencyTracker<bool>>,
    ) -> bool {
        // Inline the LLPtr → _func unwrap; on miss, fall back to the
        // upstream `BoolGraphAnalyzer.analyze_external_call` default,
        // which is `top_result() == True`.
        let Some(arg0) = op.args.first() else {
            return true;
        };
        let Hlvalue::Constant(c) = arg0 else {
            return true;
        };
        let ConstValue::LLPtr(f) = &c.value else {
            return true;
        };
        let Ok(_ptr_obj::Func(funcobj)) = f._obj() else {
            return true;
        };
        match funcobj.attrs.get("random_effects_on_gcobjs") {
            Some(ConstValue::Bool(true)) => return true,
            _ => {}
        }
        // Upstream's super-call defaults to True for an external
        // funcobj — `BoolGraphAnalyzer` inherits the no-info path
        // which never proves bottom.
        true
    }

    /// Upstream `:27-33`:
    ///
    /// ```python
    /// def analyze_simple_operation(self, op, graphinfo):
    ///     if op.opname in ('malloc', 'malloc_varsize'):
    ///         flags = op.args[1].value
    ///         return flags['flavor'] == 'gc'
    ///     else:
    ///         return (op.opname in LL_OPERATIONS and
    ///                 LL_OPERATIONS[op.opname].canmallocgc)
    /// ```
    ///
    /// Pyre stores `flags` as `ConstValue::Dict<ConstValue,
    /// ConstValue>`; the `flavor` key resolves to `ConstValue::UniStr`
    /// or `ConstValue::ByteStr` depending on the source. Both spellings
    /// are accepted to match upstream's Python dict semantics.
    fn analyze_simple_operation(&mut self, op: &SpaceOperation, _graphinfo: &()) -> bool {
        if op.opname == "malloc" || op.opname == "malloc_varsize" {
            // Upstream `:29-30 flags = op.args[1].value;
            // return flags['flavor'] == 'gc'`.
            return op
                .args
                .get(1)
                .and_then(|arg| match arg {
                    Hlvalue::Constant(c) => Some(&c.value),
                    _ => None,
                })
                .and_then(|v| match v {
                    ConstValue::Dict(d) => Some(d),
                    _ => None,
                })
                .and_then(|d| {
                    d.get(&ConstValue::UniStr("flavor".to_string()))
                        .or_else(|| d.get(&ConstValue::ByteStr(b"flavor".to_vec())))
                })
                .map(|v| {
                    matches!(
                        v,
                        ConstValue::UniStr(s) if s == "gc"
                    ) || matches!(
                        v,
                        ConstValue::ByteStr(s) if s == b"gc"
                    )
                })
                .unwrap_or(false);
        }
        // Upstream `:32-33`: opname in LL_OPERATIONS and canmallocgc.
        match ll_operations().get(op.opname.as_str()) {
            Some(desc) => desc.canmallocgc,
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{
        Block, ConstValue, Constant, FunctionGraph, GraphFunc, GraphKey, Hlvalue, SpaceOperation,
        Variable,
    };
    use crate::translator::rtyper::lltypesystem::lltype;
    use crate::translator::translator::TranslationContext;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    fn graph_with_func(name: &str) -> GraphRef {
        let start = Block::shared(vec![]);
        let mut graph = FunctionGraph::new(name, start);
        graph.func = Some(GraphFunc::new(
            name,
            Constant::new(ConstValue::Dict(HashMap::new())),
        ));
        Rc::new(RefCell::new(graph))
    }

    fn direct_call_to(graph: Option<&GraphRef>) -> SpaceOperation {
        let graph_key = graph.map(|graph| GraphKey::of(graph).as_usize());
        let ptr = lltype::functionptr(
            lltype::FuncType {
                args: Vec::new(),
                result: lltype::LowLevelType::Void,
            },
            "callee",
            graph_key,
            None,
        );
        SpaceOperation::new(
            "direct_call",
            vec![Hlvalue::Constant(Constant::new(ConstValue::LLPtr(
                Box::new(ptr),
            )))],
            Hlvalue::Variable(Variable::named("result")),
        )
    }

    /// `flags={'flavor': 'gc'}` malloc op fixture.
    fn malloc_op(flavor: &str) -> SpaceOperation {
        let mut flags = HashMap::new();
        flags.insert(
            ConstValue::UniStr("flavor".to_string()),
            ConstValue::UniStr(flavor.to_string()),
        );
        SpaceOperation::new(
            "malloc",
            vec![
                Hlvalue::Constant(Constant::new(ConstValue::None)),
                Hlvalue::Constant(Constant::new(ConstValue::Dict(flags))),
            ],
            Hlvalue::Variable(Variable::named("result")),
        )
    }

    #[test]
    fn cannot_collect_hint_short_circuits_to_false() {
        let translator = TranslationContext::new();
        let mut analyzer = CollectAnalyzer::new(&translator);
        let g = graph_with_func("g");
        g.borrow_mut()
            .func
            .as_mut()
            .unwrap()
            ._gctransformer_hint_cannot_collect_ = true;
        // Even with a body that would otherwise collect, the hint
        // forces False.
        g.borrow()
            .startblock
            .borrow_mut()
            .operations
            .push(malloc_op("gc"));
        assert!(!analyzer.analyze_direct_call(&g, None));
    }

    #[test]
    fn close_stack_hint_short_circuits_to_true() {
        let translator = TranslationContext::new();
        let mut analyzer = CollectAnalyzer::new(&translator);
        let g = graph_with_func("g");
        g.borrow_mut()
            .func
            .as_mut()
            .unwrap()
            ._gctransformer_hint_close_stack_ = true;
        // Empty body — the hint alone forces True.
        assert!(analyzer.analyze_direct_call(&g, None));
    }

    #[test]
    fn malloc_with_gc_flavor_returns_true() {
        let translator = TranslationContext::new();
        let mut analyzer = CollectAnalyzer::new(&translator);
        assert!(analyzer.analyze_simple_operation(&malloc_op("gc"), &()));
    }

    #[test]
    fn malloc_with_raw_flavor_returns_false() {
        let translator = TranslationContext::new();
        let mut analyzer = CollectAnalyzer::new(&translator);
        assert!(!analyzer.analyze_simple_operation(&malloc_op("raw"), &()));
    }

    #[test]
    fn unknown_opname_returns_false() {
        let translator = TranslationContext::new();
        let mut analyzer = CollectAnalyzer::new(&translator);
        let op = SpaceOperation::new(
            "totally_made_up_op",
            vec![],
            Hlvalue::Variable(Variable::named("r")),
        );
        assert!(!analyzer.analyze_simple_operation(&op, &()));
    }

    #[test]
    fn external_call_with_random_effects_on_gcobjs_returns_true() {
        use std::collections::HashMap as StdHashMap;
        let translator = TranslationContext::new();
        let mut analyzer = CollectAnalyzer::new(&translator);

        // Build a direct_call op with attrs = {"random_effects_on_gcobjs": True}.
        let mut attrs: StdHashMap<String, ConstValue> = StdHashMap::new();
        attrs.insert(
            "random_effects_on_gcobjs".to_string(),
            ConstValue::Bool(true),
        );
        let ptr = crate::translator::rtyper::lltypesystem::lltype::_ptr::new(
            crate::translator::rtyper::lltypesystem::lltype::Ptr {
                TO: crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Func(
                    lltype::FuncType {
                        args: vec![],
                        result: lltype::LowLevelType::Void,
                    },
                ),
            },
            Ok(Some(
                crate::translator::rtyper::lltypesystem::lltype::_ptr_obj::Func(
                    crate::translator::rtyper::lltypesystem::lltype::_func::new(
                        lltype::FuncType {
                            args: vec![],
                            result: lltype::LowLevelType::Void,
                        },
                        "ext".to_string(),
                        None,
                        None,
                        attrs,
                    ),
                ),
            )),
        );
        let op = SpaceOperation::new(
            "direct_call",
            vec![Hlvalue::Constant(Constant::new(ConstValue::LLPtr(
                Box::new(ptr),
            )))],
            Hlvalue::Variable(Variable::named("r")),
        );
        assert!(analyzer.analyze_external_call(&op, None));
    }

    #[test]
    fn external_direct_call_to_unknown_returns_true_by_default() {
        let translator = TranslationContext::new();
        let mut analyzer = CollectAnalyzer::new(&translator);
        // direct_call_to(None) builds an LLPtr whose `_func.attrs` has
        // no `random_effects_on_gcobjs` key — upstream's default in
        // BoolGraphAnalyzer is `top_result() == True`.
        let op = direct_call_to(None);
        assert!(analyzer.analyze_external_call(&op, None));
    }
}
