//! Call-family signature / annotation normalization.
//!
//! RPython upstream: `rpython/rtyper/normalizecalls.py` (414 LOC, of
//! which lines 14-204 cover the normalize_* half that this module ports).
//! The later halves (`merge_classpbc_getattr_into_classdef`,
//! `create_class_constructors`, `create_instantiate_functions`,
//! `assign_inheritance_ids`, `perform_normalizations`) land with the
//! rtyper specialization epic; they depend on `rpython/rtyper/rclass.py`
//! infrastructure that pyre still has as scaffolding only.
//!
//! ## What is ported here (upstream lines 14-204)
//!
//! - [`normalize_call_familes`] — outer loop over
//!   `bookkeeper.pbc_maximal_call_families.infos()` (upstream line 14-21).
//! - [`normalize_calltable`] — iterate shapes/rows, delegate to the
//!   row-level normalizers (upstream line 23-42).
//! - [`raise_call_table_too_complex_error`] — build TyperError message
//!   for families that span multiple shapes (upstream line 44-76).
//! - [`normalize_calltable_row_signature`] — argument-order
//!   normalization across a row (upstream line 78-154).
//! - [`normalize_calltable_row_annotation`] — annotation-union
//!   generalization across a row (upstream line 156-204).
//!
//! ## Upstream call-family clarification
//!
//! Per the user's plan correction (session transcript):
//!
//! > `normalizecalls.py`는 graph monomorphization이 아니라
//! > signature/annotation normalization만 한다.
//!
//! The job of this module is _signature agreement_ across the families
//! that the annotator has already built — e.g. two `__init__` methods on
//! different classes in the same `pbc_maximal_call_families` bucket need
//! matching argnames so the shared PBC dispatcher can call either one.
//! **It does NOT specialize generic graphs.** Specialization (what
//! actually turns `opcode_*<H>` into `opcode_*::<PyFrame>`) lives in
//! `rpython/rtyper/rpbc.py` `SingleFrozenPBCRepr` /
//! `MultipleFrozenPBCRepr` / `SmallFunctionSetPBCRepr` — that is the
//! Commit 4 target.

use crate::annotator::annrpython::RPythonAnnotator;
use crate::annotator::description::{CallFamily, CallTableRow};
use crate::annotator::model::AnnotatorError;
use crate::flowspace::argument::{CallShape, Signature};
use crate::flowspace::model::{
    Block, BlockKey, BlockRefExt, ConstValue, Constant, GraphKey, Hlvalue, Link, Variable,
    checkgraph,
};
use crate::flowspace::pygraph::PyGraph;

use std::rc::Rc;

/// RPython `normalize_call_familes(annotator)` (normalizecalls.py:14-21).
///
/// ```python
/// def normalize_call_familes(annotator):
///     for callfamily in annotator.bookkeeper.pbc_maximal_call_families.infos():
///         if not callfamily.modified:
///             assert callfamily.normalized
///             continue
///         normalize_calltable(annotator, callfamily)
///         callfamily.normalized = True
///         callfamily.modified = False
/// ```
pub fn normalize_call_familes(annotator: &RPythonAnnotator) -> Result<(), AnnotatorError> {
    // RPython: `annotator.bookkeeper.pbc_maximal_call_families.infos()`.
    // Pyre `UnionFind::infos()` yields `&Rc<RefCell<CallFamily>>`; we
    // clone the Rcs into a local Vec so the borrow on
    // `pbc_maximal_call_families` can be released before the inner
    // `normalize_calltable` call (which itself may invoke bookkeeper
    // helpers that re-borrow the UnionFind).
    let families: Vec<_> = {
        let bk = &annotator.bookkeeper;
        let pbc = bk.pbc_maximal_call_families.borrow();
        pbc.infos().cloned().collect()
    };
    for callfamily_rc in families {
        // RPython: `if not callfamily.modified: assert callfamily.normalized; continue`.
        {
            let cf = callfamily_rc.borrow();
            if !cf.modified {
                debug_assert!(
                    cf.normalized,
                    "CallFamily invariant: modified=false implies normalized=true"
                );
                continue;
            }
        }
        // RPython: `normalize_calltable(annotator, callfamily)`.
        normalize_calltable(annotator, &callfamily_rc)?;
        // RPython: `callfamily.normalized = True; callfamily.modified = False`.
        let mut cf = callfamily_rc.borrow_mut();
        cf.normalized = true;
        cf.modified = false;
    }
    Ok(())
}

/// RPython `normalize_calltable(annotator, callfamily)`
/// (normalizecalls.py:23-42).
///
/// ```python
/// def normalize_calltable(annotator, callfamily):
///     """Try to normalize all rows of a table."""
///     nshapes = len(callfamily.calltables)
///     for shape, table in callfamily.calltables.items():
///         for row in table:
///             did_something = normalize_calltable_row_signature(annotator, shape, row)
///             if did_something:
///                 assert not callfamily.normalized, "change in call family normalisation"
///                 if nshapes != 1:
///                     raise_call_table_too_complex_error(callfamily, annotator)
///     while True:
///         progress = False
///         for shape, table in callfamily.calltables.items():
///             for row in table:
///                 progress |= normalize_calltable_row_annotation(annotator, row.values())
///         if not progress:
///             return   # done
///         assert not callfamily.normalized, "change in call family normalisation"
/// ```
fn normalize_calltable(
    annotator: &RPythonAnnotator,
    callfamily_rc: &std::rc::Rc<std::cell::RefCell<CallFamily>>,
) -> Result<(), AnnotatorError> {
    // upstream line 25: `nshapes = len(callfamily.calltables)`. Clone the
    // (shape, row) pairs up front so the inner normalize_* passes can
    // receive a `&RPythonAnnotator` without the bookkeeper borrow still
    // held through the iteration.
    let (nshapes, shaped_rows): (usize, Vec<(CallShape, Vec<CallTableRow>)>) = {
        let cf = callfamily_rc.borrow();
        let pairs: Vec<_> = cf
            .calltables
            .iter()
            .map(|(shape, table)| (shape.clone(), table.clone()))
            .collect();
        (cf.calltables.len(), pairs)
    };

    // upstream line 26-33: per-row signature normalization.
    for (shape, table) in &shaped_rows {
        for row in table {
            let did_something = normalize_calltable_row_signature(annotator, shape, row)?;
            if did_something {
                // upstream line 31: `assert not callfamily.normalized, ...`.
                debug_assert!(
                    !callfamily_rc.borrow().normalized,
                    "change in call family normalisation"
                );
                if nshapes != 1 {
                    return Err(raise_call_table_too_complex_error(
                        &callfamily_rc.borrow(),
                        annotator,
                    ));
                }
            }
        }
    }

    // upstream line 34-42: `while True` fixpoint on row annotations.
    loop {
        let mut progress = false;
        // Refresh the shaped_rows snapshot each iteration — upstream
        // iterates `callfamily.calltables.items()` live, which does not
        // change during annotation normalization but may on signature
        // rewrite (row_signature can run before row_annotation in the
        // same call).
        let snapshot: Vec<_> = {
            let cf = callfamily_rc.borrow();
            cf.calltables
                .iter()
                .map(|(shape, table)| (shape.clone(), table.clone()))
                .collect()
        };
        for (_shape, table) in &snapshot {
            for row in table {
                let graphs: Vec<Rc<PyGraph>> = row.values().cloned().collect();
                progress |= normalize_calltable_row_annotation(annotator, &graphs)?;
            }
        }
        if !progress {
            return Ok(());
        }
        // upstream line 42: `assert not callfamily.normalized, ...`.
        debug_assert!(
            !callfamily_rc.borrow().normalized,
            "change in call family normalisation"
        );
    }
}

/// RPython `raise_call_table_too_complex_error(callfamily, annotator)`
/// (normalizecalls.py:44-76).
///
/// Called when a single call family carries rows with differing
/// `CallShape`s — the PBC dispatcher cannot bridge them, so the
/// annotator must stop and report the problem. Upstream raises
/// `TyperError`; pyre returns [`AnnotatorError`] carrying the same
/// multi-line message shape.
pub(crate) fn raise_call_table_too_complex_error(
    callfamily: &CallFamily,
    annotator: &RPythonAnnotator,
) -> AnnotatorError {
    let mut msg: Vec<String> = Vec::new();
    // upstream: `items = callfamily.calltables.items()` — order is
    // python-dict-insertion but the Rust HashMap does not preserve
    // that. Sort by shape to get a deterministic diagnostic output
    // that still covers every pair.
    let mut items: Vec<(&CallShape, &Vec<CallTableRow>)> = callfamily.calltables.iter().collect();
    items.sort_by(|(a, _), (b, _)| {
        a.shape_cnt
            .cmp(&b.shape_cnt)
            .then_with(|| a.shape_keys.cmp(&b.shape_keys))
            .then_with(|| a.shape_star.cmp(&b.shape_star))
    });
    for (i, (shape1, table1)) in items.iter().enumerate() {
        for (shape2, table2) in items.iter().skip(i + 1) {
            // upstream: `if shape1 == shape2: continue`.
            if shape1 == shape2 {
                continue;
            }
            // upstream: `row1 = table1[0]; row2 = table2[0]`.
            let Some(row1) = table1.first() else { continue };
            let Some(row2) = table2.first() else { continue };
            // upstream:
            //   problematic_function_graphs = set(row1.values()).union(set(row2.values()))
            //   pfg = [str(graph) for graph in problematic_function_graphs]; pfg.sort()
            let mut pfg: Vec<String> = row1
                .values()
                .chain(row2.values())
                .map(|g| g.graph.borrow().name.clone())
                .collect();
            pfg.sort();
            pfg.dedup();
            msg.push("the following functions:".to_string());
            msg.push(format!("    {}", pfg.join("\n    ")));
            msg.push("are called with inconsistent numbers of arguments".to_string());
            msg.push(
                "(and/or the argument names are different, which is not \
                 supported in this case)"
                    .to_string(),
            );
            if shape1.shape_cnt != shape2.shape_cnt {
                msg.push(format!(
                    "sometimes with {} arguments, sometimes with {}",
                    shape1.shape_cnt, shape2.shape_cnt
                ));
            }
            // upstream: `callers = []; ...` — iterate
            // `annotator.translator.callgraph.iteritems()` and collect
            // unique caller names whose callee is in
            // `problematic_function_graphs`.
            let problematic_graphs: Vec<GraphKey> = row1
                .values()
                .chain(row2.values())
                .map(|g| GraphKey::of(&g.graph))
                .collect();
            let mut callers: Vec<String> = annotator
                .translator
                .borrow()
                .callgraph
                .borrow()
                .values()
                .filter(|edge| problematic_graphs.contains(&GraphKey::of(&edge.callee)))
                .map(|edge| edge.caller.borrow().name.clone())
                .collect();
            callers.sort();
            callers.dedup();
            msg.push("the callers of these functions are:".to_string());
            for caller in callers {
                msg.push(format!("    {caller}"));
            }
        }
    }
    AnnotatorError::new(msg.join("\n"))
}

/// RPython `normalize_calltable_row_signature(annotator, shape, row)`
/// (normalizecalls.py:78-154).
///
/// Returns `true` if the row was rewritten (triggering
/// `callfamily.normalized = False` re-check upstream).
///
/// Upstream short-circuits on line 83-89 via the `for...else` idiom:
/// if every graph in the row has the same `signature` and `defaults`
/// as the first, return `False` immediately. When a graph disagrees,
/// line 91-153 rebuilds its startblock so call-family peers expose the
/// same argument order/default surface. This port now performs the
/// same rewrite.
pub(crate) fn normalize_calltable_row_signature(
    annotator: &RPythonAnnotator,
    shape: &CallShape,
    row: &CallTableRow,
) -> Result<bool, AnnotatorError> {
    let _ = annotator;
    // upstream line 79: `graphs = row.values(); assert graphs, "no graph??"`.
    let graphs: Vec<&Rc<PyGraph>> = row.values().collect();
    assert!(!graphs.is_empty(), "no graph??");

    // upstream line 81-89: fast path — all signatures + defaults match.
    let sig0 = graphs[0].signature.borrow().clone();
    let defaults0 = graphs[0].defaults.borrow().clone();
    let all_match = graphs
        .iter()
        .skip(1)
        .all(|g| *g.signature.borrow() == sig0 && *g.defaults.borrow() == defaults0);
    if all_match {
        // upstream line 89: `return False`.
        return Ok(false);
    }

    // upstream line 91-92: `shape_cnt, shape_keys, shape_star = shape;
    // assert not shape_star`.
    assert!(
        !shape.shape_star,
        "shape_star should have been removed at this stage"
    );

    let call_nbargs = shape.shape_cnt + shape.shape_keys.len();
    let mut did_something = false;

    for graph in graphs {
        let signature = graph.signature.borrow().clone();
        let argnames = signature.argnames;
        assert!(
            signature.varargname.is_none(),
            "normalize_calltable_row_signature: vararg not implemented"
        );
        assert!(
            signature.kwargname.is_none(),
            "normalize_calltable_row_signature: kwarg not implemented"
        );

        let graph_args = graph.graph.borrow().getargs();
        let inputargs_s: Vec<_> = graph_args.iter().map(|v| annotator.binding(v)).collect();
        let mut argorder: Vec<usize> = (0..shape.shape_cnt).collect();
        for key in &shape.shape_keys {
            let i = argnames
                .iter()
                .position(|name| name == key)
                .ok_or_else(|| {
                    AnnotatorError::new(format!(
                        "normalize_calltable_row_signature: arg {key:?} not found in graph"
                    ))
                })?;
            assert!(!argorder.contains(&i));
            argorder.push(i);
        }
        let need_reordering = argorder != (0..call_nbargs).collect::<Vec<_>>();
        if !need_reordering && graph_args.len() == call_nbargs {
            continue;
        }

        let oldblock = graph.graph.borrow().startblock.clone();
        let old_annotated = annotator
            .annotated
            .borrow()
            .get(&BlockKey::of(&oldblock))
            .cloned()
            .unwrap_or(None);
        let defaults = graph.defaults.borrow().clone().unwrap_or_default();
        let num_nondefaults = inputargs_s.len().saturating_sub(defaults.len());
        let mut padded_defaults = vec![Constant::new(ConstValue::Placeholder); num_nondefaults];
        padded_defaults.extend(defaults);
        let mut newdefaults: Vec<Constant> = Vec::new();
        let mut inlist: Vec<Hlvalue> = Vec::new();

        for &j in &argorder {
            let mut v = match &graph_args[j] {
                Hlvalue::Variable(v) => {
                    let mut fresh = Variable::new();
                    fresh.set_name_from(v);
                    fresh
                }
                Hlvalue::Constant(_) => {
                    return Err(AnnotatorError::new(
                        "normalize_calltable_row_signature: expected Variable arg",
                    ));
                }
            };
            annotator.setbinding(&mut v, inputargs_s[j].clone());
            inlist.push(Hlvalue::Variable(v));
            newdefaults.push(padded_defaults[j].clone());
        }

        let newblock = Block::shared(inlist.clone());
        let mut outlist = inlist[..shape.shape_cnt].to_vec();
        for j in shape.shape_cnt..inputargs_s.len() {
            if let Some(i) = argorder.iter().position(|idx| *idx == j) {
                outlist.push(inlist[i].clone());
                continue;
            }
            let default = padded_defaults[j].clone();
            if matches!(default.value, ConstValue::Placeholder) {
                return Err(AnnotatorError::new(format!(
                    "call pattern has {} positional arguments, but {:?} takes at least {} arguments",
                    shape.shape_cnt,
                    graph.graph.borrow().name,
                    num_nondefaults
                )));
            }
            outlist.push(Hlvalue::Constant(default));
        }

        let link = Rc::new(std::cell::RefCell::new(Link::new(
            outlist,
            Some(oldblock.clone()),
            None,
        )));
        newblock.closeblock(vec![link]);
        graph.graph.borrow_mut().startblock = newblock.clone();

        let mut trimmed_defaults = newdefaults;
        for i in (0..trimmed_defaults.len()).rev() {
            if matches!(trimmed_defaults[i].value, ConstValue::Placeholder) {
                trimmed_defaults = trimmed_defaults[i..].to_vec();
                break;
            }
        }
        *graph.signature.borrow_mut() = Signature::new(
            argorder.iter().map(|&j| argnames[j].clone()).collect(),
            None,
            None,
        );
        *graph.defaults.borrow_mut() = Some(trimmed_defaults);

        checkgraph(&graph.graph.borrow());
        let new_key = BlockKey::of(&newblock);
        annotator
            .annotated
            .borrow_mut()
            .insert(new_key.clone(), old_annotated.clone());
        annotator.all_blocks.borrow_mut().insert(new_key, newblock);
        did_something = true;
    }

    Ok(did_something)
}

/// RPython `normalize_calltable_row_annotation(annotator, graphs)`
/// (normalizecalls.py:156-204).
///
/// Returns `true` if any graph's argument or return binding was
/// widened (triggering the outer `while True` to re-run).
///
/// Upstream line 180-198 writes a replacement startblock for every
/// graph whose bindings lost information relative to the union. This
/// port now performs the same graph rewrite and returnvar widening.
pub(crate) fn normalize_calltable_row_annotation(
    annotator: &RPythonAnnotator,
    graphs: &[Rc<PyGraph>],
) -> Result<bool, AnnotatorError> {
    use crate::annotator::model::unionof;

    // upstream line 157: `if len(graphs) <= 1: return False`.
    if graphs.len() <= 1 {
        return Ok(false);
    }

    // upstream line 159-162:
    //   graph_bindings = {}
    //   for graph in graphs:
    //       graph_bindings[graph] = [annotator.binding(v) for v in graph.getargs()]
    let graph_bindings: Vec<Vec<_>> = graphs
        .iter()
        .map(|g| {
            g.graph
                .borrow()
                .getargs()
                .iter()
                .map(|v| annotator.binding(v))
                .collect()
        })
        .collect();

    // upstream line 163-166: `nbargs = len(...); assert len(binding) == nbargs`.
    let nbargs = graph_bindings[0].len();
    for binding in &graph_bindings {
        assert_eq!(binding.len(), nbargs, "inconsistent arg counts in row");
    }

    // upstream line 168-174: `generalizedargs` is the union of each
    // column. Pyre's `unionof` is N-ary (accepts any iterator) and
    // returns `Result<SomeValue, UnionError>` — the `UnionError → AnnotatorError`
    // bridge is spelled explicitly because there is no blanket `From`
    // impl between the two (UnionError is annotator-internal).
    let mut generalizedargs = Vec::with_capacity(nbargs);
    for i in 0..nbargs {
        let column: Vec<&_> = graph_bindings.iter().map(|b| &b[i]).collect();
        let s_value =
            unionof(column.into_iter()).map_err(|e| AnnotatorError::new(e.to_string()))?;
        generalizedargs.push(s_value);
    }

    // upstream line 175-177: `generalizedresult = unionof(*result_s)`.
    let result_s: Vec<_> = graphs
        .iter()
        .map(|g| annotator.binding(&g.graph.borrow().getreturnvar()))
        .collect();
    let generalizedresult =
        unionof(result_s.iter()).map_err(|e| AnnotatorError::new(e.to_string()))?;

    // upstream line 179-204: `conversion = False; for graph in graphs: ...`.
    let mut conversion = false;
    for (graph, bindings) in graphs.iter().zip(&graph_bindings) {
        if *bindings != generalizedargs {
            conversion = true;
            let oldblock = graph.graph.borrow().startblock.clone();
            let old_annotated = annotator
                .annotated
                .borrow()
                .get(&BlockKey::of(&oldblock))
                .cloned()
                .unwrap_or(None);
            let graph_args = graph.graph.borrow().getargs();
            let mut inlist = Vec::with_capacity(generalizedargs.len());
            for (j, s_value) in generalizedargs.iter().enumerate() {
                let mut v = match &graph_args[j] {
                    Hlvalue::Variable(v) => {
                        let mut fresh = Variable::new();
                        fresh.set_name_from(v);
                        fresh
                    }
                    Hlvalue::Constant(_) => {
                        return Err(AnnotatorError::new(
                            "normalize_calltable_row_annotation: expected Variable arg",
                        ));
                    }
                };
                annotator.setbinding(&mut v, s_value.clone());
                inlist.push(Hlvalue::Variable(v));
            }
            let newblock = Block::shared(inlist.clone());
            let link = Rc::new(std::cell::RefCell::new(Link::new(
                inlist,
                Some(oldblock.clone()),
                None,
            )));
            newblock.closeblock(vec![link]);
            graph.graph.borrow_mut().startblock = newblock.clone();
            checkgraph(&graph.graph.borrow());
            let new_key = BlockKey::of(&newblock);
            annotator
                .annotated
                .borrow_mut()
                .insert(new_key.clone(), old_annotated.clone());
            annotator.all_blocks.borrow_mut().insert(new_key, newblock);
        }
        let fg = graph.graph.borrow_mut();
        if let Hlvalue::Variable(v) = &mut fg.returnblock.borrow_mut().inputargs[0] {
            if annotator.binding(&Hlvalue::Variable(v.clone())) != generalizedresult {
                conversion = true;
                annotator.setbinding(v, generalizedresult.clone());
            }
        }
    }

    Ok(conversion)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::description::DescKey;
    use crate::annotator::model::{SomeInteger, SomeValue};
    use crate::flowspace::model::{FunctionGraph, GraphFunc};

    fn empty_shape() -> CallShape {
        CallShape {
            shape_cnt: 0,
            shape_keys: Vec::new(),
            shape_star: false,
        }
    }

    fn make_pygraph(name: &str, argnames: &[&str], defaults: Option<Vec<Constant>>) -> Rc<PyGraph> {
        let inputargs: Vec<_> = argnames
            .iter()
            .map(|name| Hlvalue::Variable(Variable::named(*name)))
            .collect();
        let startblock = Block::shared(inputargs);
        let graph = FunctionGraph::new(name, startblock.clone());
        let ret_arg = graph.startblock.borrow().inputargs[0].clone();
        let link = Rc::new(std::cell::RefCell::new(Link::new(
            vec![ret_arg],
            Some(graph.returnblock.clone()),
            None,
        )));
        startblock.closeblock(vec![link]);
        Rc::new(PyGraph {
            graph: Rc::new(std::cell::RefCell::new(graph)),
            func: GraphFunc::new(name, Constant::new(ConstValue::Dict(Default::default()))),
            signature: std::cell::RefCell::new(Signature::new(
                argnames.iter().map(|s| (*s).to_string()).collect(),
                None,
                None,
            )),
            defaults: std::cell::RefCell::new(defaults),
            access_directly: std::cell::Cell::new(false),
        })
    }

    fn bind_graph_inputs(
        ann: &RPythonAnnotator,
        graph: &Rc<PyGraph>,
        inputs: &[SomeValue],
        result: SomeValue,
    ) {
        {
            let startblock = graph.graph.borrow().startblock.clone();
            let mut blk = startblock.borrow_mut();
            for (arg, value) in blk.inputargs.iter_mut().zip(inputs.iter()) {
                let Hlvalue::Variable(v) = arg else {
                    panic!("expected variable input");
                };
                ann.setbinding(v, value.clone());
            }
        }
        {
            let returnblock = graph.graph.borrow().returnblock.clone();
            let mut blk = returnblock.borrow_mut();
            let Hlvalue::Variable(v) = &mut blk.inputargs[0] else {
                panic!("expected variable return");
            };
            ann.setbinding(v, result);
        }
        let startblock = graph.graph.borrow().startblock.clone();
        let graphref = graph.graph.clone();
        let bkey = BlockKey::of(&startblock);
        ann.annotated
            .borrow_mut()
            .insert(bkey.clone(), Some(graphref));
        ann.all_blocks.borrow_mut().insert(bkey, startblock);
    }

    /// `raise_call_table_too_complex_error` is the only pure function in
    /// this module that does not require an `RPythonAnnotator`. Build a
    /// minimal `CallFamily` with two shapes and verify the message
    /// payload carries the upstream-style multi-line structure.
    #[test]
    fn raise_error_with_two_shapes_lists_inconsistent_arity() {
        let fake_desc = DescKey(1);
        let mut fam = CallFamily::new(fake_desc);
        // Inject two empty tables under distinct shapes so the
        // enumerate-cross loop runs once and produces at least the
        // inconsistent-arity message.
        fam.calltables
            .insert(empty_shape(), vec![CallTableRow::new()]);
        let mut shape2 = empty_shape();
        shape2.shape_cnt = 1;
        fam.calltables.insert(shape2, vec![CallTableRow::new()]);
        // Build a throwaway RPythonAnnotator via the public `new()`
        // helper to satisfy the signature; the function does not
        // dereference it (the callgraph lookup is deferred).
        let ann = RPythonAnnotator::new(None, None, None, false);
        let err = raise_call_table_too_complex_error(&fam, &ann);
        let msg = err.to_string();
        assert!(
            msg.contains("are called with inconsistent numbers of arguments"),
            "missing upstream phrase; got: {msg}"
        );
        assert!(
            msg.contains("sometimes with 0 arguments, sometimes with 1")
                || msg.contains("sometimes with 1 arguments, sometimes with 0"),
            "missing shape-cnt diff hint; got: {msg}"
        );
    }

    #[test]
    fn raise_error_lists_callers_from_translator_callgraph() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let callee = make_pygraph("callee", &["x"], None);
        let caller = Rc::new(std::cell::RefCell::new(FunctionGraph::new(
            "caller",
            Block::shared(vec![]),
        )));
        let caller_block = caller.borrow().startblock.clone();
        ann.translator.borrow().update_call_graph(
            &caller,
            &callee.graph,
            (BlockKey::of(&caller_block), 0),
        );

        let fake_desc = DescKey(1);
        let mut fam = CallFamily::new(fake_desc);
        let mut row1 = CallTableRow::new();
        row1.insert(DescKey(2), callee.clone());
        fam.calltables.insert(empty_shape(), vec![row1]);
        let mut row2 = CallTableRow::new();
        row2.insert(DescKey(3), callee);
        let mut shape2 = empty_shape();
        shape2.shape_cnt = 1;
        fam.calltables.insert(shape2, vec![row2]);

        let err = raise_call_table_too_complex_error(&fam, &ann);
        let msg = err.to_string();
        assert!(msg.contains("the callers of these functions are:"));
        assert!(msg.contains("    caller"));
    }

    #[test]
    fn row_signature_rewrites_startblock_and_signature() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = make_pygraph(
            "f1",
            &["a", "b", "c"],
            Some(vec![
                Constant::new(ConstValue::Int(10)),
                Constant::new(ConstValue::Int(20)),
            ]),
        );
        let graph2 = make_pygraph(
            "f2",
            &["a", "c"],
            Some(vec![Constant::new(ConstValue::Int(20))]),
        );
        bind_graph_inputs(
            &ann,
            &graph,
            &[
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::Integer(SomeInteger::default()),
            ],
            SomeValue::Integer(SomeInteger::default()),
        );
        bind_graph_inputs(
            &ann,
            &graph2,
            &[
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::Integer(SomeInteger::default()),
            ],
            SomeValue::Integer(SomeInteger::default()),
        );
        let mut row = CallTableRow::new();
        row.insert(DescKey(1), graph.clone());
        row.insert(DescKey(2), graph2);
        let shape = CallShape {
            shape_cnt: 1,
            shape_keys: vec!["c".to_string()],
            shape_star: false,
        };

        assert!(normalize_calltable_row_signature(&ann, &shape, &row).unwrap());
        assert_eq!(
            graph.signature.borrow().argnames,
            vec!["a".to_string(), "c".to_string()]
        );
        assert_eq!(graph.graph.borrow().startblock.borrow().inputargs.len(), 2);
        let graph_ref = graph.graph.borrow();
        let startblock = graph_ref.startblock.clone();
        let exits_len = startblock.borrow().exits.len();
        assert_eq!(exits_len, 1);
        assert_eq!(startblock.borrow().exits[0].borrow().args.len(), 3);
    }

    #[test]
    fn row_annotation_rewrites_inputs_and_widens_return() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph_int = make_pygraph("g1", &["x"], None);
        let graph_bool = make_pygraph("g2", &["x"], None);
        bind_graph_inputs(
            &ann,
            &graph_int,
            &[SomeValue::Integer(SomeInteger::default())],
            SomeValue::Integer(SomeInteger::default()),
        );
        bind_graph_inputs(
            &ann,
            &graph_bool,
            &[crate::annotator::model::s_bool()],
            crate::annotator::model::s_bool(),
        );

        assert!(
            normalize_calltable_row_annotation(&ann, &[graph_int.clone(), graph_bool.clone()])
                .unwrap()
        );

        let startblock = graph_int.graph.borrow().startblock.clone();
        let arg = startblock.borrow().inputargs[0].clone();
        assert!(matches!(ann.binding(&arg), SomeValue::Integer(_)));
        let ret = graph_int.graph.borrow().getreturnvar();
        assert!(matches!(ann.binding(&ret), SomeValue::Integer(_)));
    }
}
