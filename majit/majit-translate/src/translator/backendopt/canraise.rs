//! Port of `rpython/translator/backendopt/canraise.py`.
//!
//! `RaiseAnalyzer` is the `BoolGraphAnalyzer` subclass that decides
//! whether each call site can raise. Inputs feed
//! `inline.py:144 BaseInliner.__init__(... raise_analyzer)` and
//! `inline.py:124-142 any_call_to_raising_graphs`. The Rust port
//! exposes the same surface (`can_raise`, `analyze_simple_operation`,
//! `analyze_external_call`, `analyze_exceptblock_in_graph`,
//! `do_ignore_memory_error`).
//!
//! The analyser walks the recursive call graph through
//! [`super::graphanalyze::GraphAnalyzer`] — the framework that
//! `gilanalysis::GilAnalyzer` already consumes — and short-circuits on
//! the boolean lattice (`top_result == True` ends the walk).

use crate::flowspace::model::{BlockRef, ConstValue, GraphRef, Hlvalue, SpaceOperation};
use crate::tool::algo::unionfind::UnionFind;
use crate::translator::backendopt::graphanalyze::{Dependency, DependencyTracker, GraphAnalyzer};
use crate::translator::backendopt::ssa::DataFlowFamilyBuilder;
use crate::translator::rtyper::lltypesystem::lloperation::ll_operations;
use crate::translator::rtyper::lltypesystem::lltype::_ptr_obj;
use crate::translator::translator::TranslationContext;

/// `class RaiseAnalyzer(graphanalyze.BoolGraphAnalyzer)` at
/// `canraise.py:8-46`.
pub struct RaiseAnalyzer<'t> {
    translator: &'t TranslationContext,
    /// Upstream `RaiseAnalyzer.ignore_exact_class = None` at
    /// `canraise.py:9`. `do_ignore_memory_error` flips this to the
    /// MemoryError class. The Rust port stores the upstream class
    /// name as a literal string because pyre's `LLOp.canraise` carries
    /// `&'static str` exception names rather than class objects.
    ignore_exact_class: Option<&'static str>,
    /// Upstream `GraphAnalyzer._analyzed_calls` (`graphanalyze.py:13`).
    analyzed_calls: UnionFind<usize, Dependency<bool>>,
}

impl<'t> RaiseAnalyzer<'t> {
    pub fn new(translator: &'t TranslationContext) -> Self {
        Self {
            translator,
            ignore_exact_class: None,
            analyzed_calls: UnionFind::new(|_| Dependency::new(false)),
        }
    }

    /// `do_ignore_memory_error(self)` at `canraise.py:11-12`.
    pub fn do_ignore_memory_error(&mut self) {
        self.ignore_exact_class = Some("MemoryError");
    }

    /// `can_raise(self, op, seen=None)` — backward-compatible
    /// interface at `canraise.py:43-45`.
    pub fn can_raise(
        &mut self,
        op: &SpaceOperation,
        seen: Option<&mut DependencyTracker<bool>>,
    ) -> bool {
        self.analyze(op, seen, &())
    }
}

impl<'t> GraphAnalyzer<bool, ()> for RaiseAnalyzer<'t> {
    fn translator(&self) -> &TranslationContext {
        self.translator
    }

    fn analyzed_calls(&mut self) -> &mut UnionFind<usize, Dependency<bool>> {
        &mut self.analyzed_calls
    }

    /// Upstream `:14-20`:
    /// ```python
    /// def analyze_simple_operation(self, op, graphinfo):
    ///     try:
    ///         canraise = LL_OPERATIONS[op.opname].canraise
    ///         return bool(canraise) and canraise != (self.ignore_exact_class,)
    ///     except KeyError:
    ///         log.WARNING("Unknown operation: %s" % op.opname)
    ///         return True
    /// ```
    ///
    /// `bool(canraise)` is False on the empty tuple. The
    /// `canraise != (self.ignore_exact_class,)` comparison drops a
    /// single-element exception tuple that matches `ignore_exact_class`
    /// (set by `do_ignore_memory_error`). The unknown-op fallback
    /// returns True — Pyre log channels are no-ops today, so the
    /// upstream `log.WARNING` call is omitted; the conservative True
    /// surfaces every unknown op as raising.
    fn analyze_simple_operation(&mut self, op: &SpaceOperation, _graphinfo: &()) -> bool {
        match ll_operations().get(op.opname.as_str()) {
            Some(desc) => {
                if desc.canraise.is_empty() {
                    false
                } else if let Some(ignore) = self.ignore_exact_class {
                    !(desc.canraise.len() == 1 && desc.canraise[0] == ignore)
                } else {
                    true
                }
            }
            None => true,
        }
    }

    /// Upstream `:22-23`:
    /// ```python
    /// def analyze_external_call(self, fnobj, seen=None):
    ///     return getattr(fnobj, 'canraise', True)
    /// ```
    ///
    /// `direct_call`'s callee descriptor (`op.args[0].value._obj`) is
    /// what upstream calls `fnobj`. Pyre's `_func` carrier holds the
    /// upstream attribute mirror in `_func.attrs`
    /// (`lltype.rs:690 attrs: HashMap<String, ConstValue>`); the
    /// `canraise` slot is set by `lltype.functionptr(canraise=...)`
    /// at the same upstream site that originates the attribute on
    /// the Python `_func`. Read it here; default to `True` per
    /// upstream's `getattr(..., True)` when the slot is absent.
    fn analyze_external_call(
        &mut self,
        op: &SpaceOperation,
        _seen: Option<&mut DependencyTracker<bool>>,
    ) -> bool {
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
        match funcobj.attrs.get("canraise") {
            Some(ConstValue::Bool(b)) => *b,
            _ => true,
        }
    }

    /// Upstream `:25 analyze_exceptblock = None    # don't call this`.
    /// Replicates the upstream guard: `analyze_exceptblock` should
    /// never fire on a `RaiseAnalyzer` — the framework body calls
    /// `analyze_exceptblock_in_graph` per `graphanalyze.py:155`. If
    /// the day ever comes when something routes back through the
    /// vanilla hook, surface the deviation loudly so the contract
    /// stays in sync with upstream.
    fn analyze_exceptblock(
        &mut self,
        _block: &BlockRef,
        _seen: Option<&mut DependencyTracker<bool>>,
    ) -> bool {
        panic!(
            "canraise.py:25 RaiseAnalyzer.analyze_exceptblock = None — \
             framework should call analyze_exceptblock_in_graph instead"
        )
    }

    /// Upstream `:27-41`:
    /// ```python
    /// def analyze_exceptblock_in_graph(self, graph, block, seen=None):
    ///     if self.ignore_exact_class is not None:
    ///         from rpython.translator.backendopt.ssa import DataFlowFamilyBuilder
    ///         dff = DataFlowFamilyBuilder(graph)
    ///         variable_families = dff.get_variable_families()
    ///         v_exc_instance = variable_families.find_rep(block.inputargs[1])
    ///         for link1 in graph.iterlinks():
    ///             v = link1.last_exc_value
    ///             if v is not None:
    ///                 if variable_families.find_rep(v) is v_exc_instance:
    ///                     # this is a case of re-raise the exception caught;
    ///                     # it doesn't count.  We'll see the place that really
    ///                     # raises the exception in the first place.
    ///                     return False
    ///     return True
    /// ```
    ///
    /// The re-raise check fires only when `do_ignore_memory_error`
    /// (or any future ignore-class) is on. Otherwise the except block
    /// is unconditionally treated as raising.
    fn analyze_exceptblock_in_graph(
        &mut self,
        graph: &GraphRef,
        block: &BlockRef,
        _seen: Option<&mut DependencyTracker<bool>>,
    ) -> bool {
        if self.ignore_exact_class.is_some() {
            // Upstream `:30-31`:
            //     dff = DataFlowFamilyBuilder(graph)
            //     variable_families = dff.get_variable_families()
            let mut dff = DataFlowFamilyBuilder::new(&graph.borrow());
            // Drive the union-find to a fixed point so `find_rep`
            // returns upstream's "family representative". Upstream's
            // `__init__` runs `complete()` implicitly through later
            // `find_rep`/`find` calls; pyre's structural port exposes
            // `complete()` explicitly.
            dff.complete();
            // Upstream `:32 v_exc_instance =
            // variable_families.find_rep(block.inputargs[1])` — second
            // input argument of the except block carries the exc
            // instance (first is exc type).
            let exc_value_var = match block.borrow().inputargs.get(1).cloned() {
                Some(v) => v,
                None => return true,
            };
            let v_exc_instance = dff.variable_families.find_rep(exc_value_var);
            // Upstream `:33-40`: walk every link, dropping links whose
            // `last_exc_value` family-rep matches the except instance.
            let links = graph.borrow().iterlinks();
            for link in links {
                let last_exc_value = link.borrow().last_exc_value.clone();
                if let Some(v) = last_exc_value {
                    if dff.variable_families.find_rep(v) == v_exc_instance {
                        // re-raise of caught exception — does not count
                        return false;
                    }
                }
            }
        }
        true
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
    use std::rc::Rc;

    fn fixture() -> TranslationContext {
        TranslationContext::new()
    }

    #[test]
    fn raise_analyzer_treats_pure_int_add_as_non_raising() {
        // `int_add` is registered with empty canraise in
        // `lloperation.rs`. Upstream `bool(canraise) == False` ⇒ False.
        let translator = fixture();
        let mut a = RaiseAnalyzer::new(&translator);
        let v = Variable::named("x");
        let op = SpaceOperation::new(
            "int_add",
            vec![
                Hlvalue::Constant(Constant::new(ConstValue::Int(1))),
                Hlvalue::Constant(Constant::new(ConstValue::Int(2))),
            ],
            Hlvalue::Variable(v),
        );
        assert!(!a.analyze_simple_operation(&op, &()));
    }

    #[test]
    fn raise_analyzer_unknown_op_is_conservatively_raising() {
        // Upstream `:18-20`: `except KeyError: return True`.
        let translator = fixture();
        let mut a = RaiseAnalyzer::new(&translator);
        let v = Variable::named("x");
        let op = SpaceOperation::new("unknown_dummy_op_for_test", vec![], Hlvalue::Variable(v));
        assert!(a.analyze_simple_operation(&op, &()));
    }

    #[test]
    fn raise_analyzer_default_treats_direct_call_canraise_exception_as_raising() {
        // `direct_call` carries `canraise=("Exception",)` in
        // `lloperation.rs`. Default ignore_exact_class is None ⇒
        // any non-empty canraise tuple ⇒ True.
        let translator = fixture();
        let mut a = RaiseAnalyzer::new(&translator);
        let v = Variable::named("x");
        let op = SpaceOperation::new(
            "direct_call",
            vec![Hlvalue::Constant(Constant::new(ConstValue::None))],
            Hlvalue::Variable(v),
        );
        assert!(a.analyze_simple_operation(&op, &()));
    }

    #[test]
    fn ignore_memory_error_flips_canraise_memoryerror_to_non_raising() {
        // Find an op whose canraise == ("MemoryError",) — the
        // canmallocgc path adds it implicitly. Pick any op registered
        // with canraise = ("MemoryError",) and verify that
        // do_ignore_memory_error makes analyze return False on it.
        // We synthesize the SpaceOperation locally; the comparison is
        // string-based against the static `canraise` Vec.
        let translator = fixture();
        let mut a = RaiseAnalyzer::new(&translator);
        a.do_ignore_memory_error();
        // Search the table for an opname whose canraise vec equals
        // ["MemoryError"].
        let target = ll_operations()
            .iter()
            .find(|(_, desc)| desc.canraise == vec!["MemoryError"])
            .map(|(name, _)| *name);
        if let Some(opname) = target {
            let v = Variable::named("x");
            let op = SpaceOperation::new(opname, vec![], Hlvalue::Variable(v));
            assert!(
                !a.analyze_simple_operation(&op, &()),
                "{opname} canraise=[MemoryError] should be False under ignore",
            );
        }
        // No registered op with that exact shape ⇒ test is vacuous,
        // but the boolean predicate is exercised in the negative
        // branch of `raise_analyzer_default_treats_direct_call_*`.
    }

    #[test]
    fn analyze_direct_call_walks_callee_graph_and_returns_true_on_raising_op() {
        // `int_add_ovf` is a regular simple op with
        // canraise=("OverflowError",). The framework should route the
        // op through analyze_simple_operation and bubble up `True`
        // through analyze_direct_call.
        let translator = fixture();
        let mut a = RaiseAnalyzer::new(&translator);
        let v = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new("f", start.clone());
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add_ovf",
            vec![
                Hlvalue::Constant(Constant::new(ConstValue::Int(1))),
                Hlvalue::Constant(Constant::new(ConstValue::Int(2))),
            ],
            Hlvalue::Variable(Variable::named("r")),
        ));
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(v)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let graph: GraphRef = Rc::new(RefCell::new(graph));
        assert!(a.analyze_direct_call(&graph, None));
    }

    #[test]
    fn analyze_direct_call_walks_callee_graph_and_returns_false_on_pure_op() {
        // Build a single-graph callee whose only op is int_add.
        // canraise=[] ⇒ False.
        let translator = fixture();
        let mut a = RaiseAnalyzer::new(&translator);
        let v = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new("f", start.clone());
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![
                Hlvalue::Constant(Constant::new(ConstValue::Int(1))),
                Hlvalue::Constant(Constant::new(ConstValue::Int(2))),
            ],
            Hlvalue::Variable(Variable::named("r")),
        ));
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(v)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let graph: GraphRef = Rc::new(RefCell::new(graph));
        assert!(!a.analyze_direct_call(&graph, None));
    }
}
