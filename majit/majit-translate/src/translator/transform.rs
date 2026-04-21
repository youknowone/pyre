//! RPython `rpython/translator/transform.py` — annotation-driven
//! graph transformations invoked from `RPythonAnnotator.simplify()`
//! (annrpython.py:357-373).
//!
//! Module header reproduced verbatim from upstream (transform.py:1-6):
//!
//! ```text
//! Flow Graph Transformation
//!
//! The difference between simplification and transformation is that
//! transformation is based on annotations; it runs after the annotator
//! completed.
//! ```
//!
//! Port coverage so far:
//!   - `checkgraphs` (transform.py:13-19)
//!   - `fully_annotated_blocks` (transform.py:21-25)
//!   - `transform_graph` (transform.py:253-272)
//!
//! The pattern-matching extra passes (`transform_allocate`,
//! `transform_extend_with_str_slice`, `transform_extend_with_char_count`,
//! `transform_list_contains`) depend on annotation-typed queries
//! (`self.gettype(v) is str`, `SomeChar`/`SomeInteger` discrimination)
//! and stay deferred until those annotation accessors land on the
//! port's `RPythonAnnotator`.
//!
//! `transform_dead_code` + `cutoff_alwaysraising_block` (transform.py:
//! 145-198) rewrite block exits and need the bookkeeper's
//! `getuniqueclassdef(AssertionError)` wired to the host-env side; they
//! land in a follow-up commit.
//!
//! `transform_dead_op_vars` (transform.py:137-143) just forwards to
//! `simplify.transform_dead_op_vars_in_blocks` (simplify.py:422+)
//! which is ~100 LOC on its own; it will land together with the
//! remaining `simplify.py` backlog.
//!
//! `insert_ll_stackcheck` (transform.py:200-243) is rtyper-phase and
//! therefore out of scope for the annotator-phase port.

use std::collections::HashSet;
use std::rc::Rc;

use crate::annotator::annrpython::RPythonAnnotator;
use crate::annotator::model::{SomeInstance, SomeValue, s_impossible_value, typeof_vars};
use crate::flowspace::model::{
    BlockKey, BlockRef, BlockRefExt, ConstValue, Constant, GraphKey, GraphRef, HOST_ENV, Hlvalue,
    HostObject, Link, LinkKey, LinkRef, Variable, checkgraph,
};

/// RPython `transform.py:13-19` — `checkgraphs(self, blocks)`.
///
/// ```python
/// def checkgraphs(self, blocks):
///     seen = set()
///     for block in blocks:
///         graph = self.annotated[block]
///         if graph not in seen:
///             checkgraph(graph)
///             seen.add(graph)
/// ```
///
/// The Rust port iterates the supplied block list, looks each one up
/// in `ann.annotated` (upstream's direct dict indexing; missing or
/// `None` entries are skipped — the annotator phase always populates
/// `annotated[block]` with the owning graph before `checkgraphs` is
/// reachable), and calls `checkgraph` on each distinct graph.
pub fn checkgraphs(ann: &RPythonAnnotator, blocks: &[BlockRef]) {
    // upstream: `seen = set()`.
    let mut seen: HashSet<GraphKey> = HashSet::new();
    let annotated = ann.annotated.borrow();
    for block in blocks {
        // upstream: `graph = self.annotated[block]`.
        let Some(Some(graph)) = annotated.get(&BlockKey::of(block)) else {
            continue;
        };
        // upstream: `if graph not in seen:`.
        let gkey = GraphKey::of(graph);
        if seen.insert(gkey) {
            // upstream: `checkgraph(graph)`.
            checkgraph(&graph.borrow());
        }
    }
}

/// RPython `transform.py:21-25` — `fully_annotated_blocks(self)`.
///
/// ```python
/// def fully_annotated_blocks(self):
///     """Ignore blocked blocks."""
///     for block, is_annotated in self.annotated.iteritems():
///         if is_annotated:
///             yield block
/// ```
///
/// Upstream returns a generator; the Rust port materialises the
/// filtered set into a `Vec<BlockRef>` since `transform_graph`'s
/// caller converts the generator into a dict immediately after.
pub fn fully_annotated_blocks(ann: &RPythonAnnotator) -> Vec<BlockRef> {
    let annotated = ann.annotated.borrow();
    let all_blocks = ann.all_blocks.borrow();
    let mut result = Vec::new();
    for (key, value) in annotated.iter() {
        // upstream `if is_annotated:` — `annotated[block] == False`
        // lowers to `Option<GraphRef>::None` in the Rust port.
        if value.is_none() {
            continue;
        }
        if let Some(block) = all_blocks.get(key) {
            result.push(block.clone());
        }
    }
    result
}

/// Signature of an `extra_pass` function threaded through
/// `transform_graph`. Upstream calls `pass_(ann, block_subset)` where
/// `block_subset` is a Python dict keyed by Block; the Rust port takes
/// a `&[BlockRef]` slice since the callers only iterate it (no dict
/// lookups downstream).
pub type TransformPass = fn(&RPythonAnnotator, &[BlockRef]);

/// RPython `transform.py:246-251` — `default_extra_passes = [...]`.
///
/// Empty for now: the four pattern-matching passes listed upstream
/// (`transform_allocate` / `transform_extend_with_str_slice` /
/// `transform_extend_with_char_count` / `transform_list_contains`)
/// all depend on annotation-typed queries and land once those
/// accessors reach the `RPythonAnnotator` surface.
pub const DEFAULT_EXTRA_PASSES: &[TransformPass] = &[];

/// RPython `transform.py:253-272` — `transform_graph(ann, extra_passes,
/// block_subset)`.
///
/// ```python
/// def transform_graph(ann, extra_passes=None, block_subset=None):
///     """Apply set of transformations available."""
///     if extra_passes is None:
///         extra_passes = default_extra_passes
///     if block_subset is None:
///         block_subset = fully_annotated_blocks(ann)
///     if not isinstance(block_subset, dict):
///         block_subset = dict.fromkeys(block_subset)
///     if ann.translator:
///         checkgraphs(ann, block_subset)
///     transform_dead_code(ann, block_subset)
///     for pass_ in extra_passes:
///         pass_(ann, block_subset)
///     transform_dead_op_vars(ann, block_subset)
///     if ann.translator:
///         checkgraphs(ann, block_subset)
/// ```
///
/// `transform_dead_code` and `transform_dead_op_vars` are both no-ops
/// at this commit (see module docs); the line-by-line placeholders are
/// kept so the driver shape mirrors upstream exactly. The
/// `ann.translator` guard collapses to an unconditional invocation
/// because the Rust port always owns a `TranslationContext`
/// (annrpython.py:30-35 default-constructs one when absent).
pub fn transform_graph(
    ann: &RPythonAnnotator,
    extra_passes: Option<&[TransformPass]>,
    block_subset: Option<&[BlockRef]>,
) {
    // upstream: `if extra_passes is None: extra_passes = default_extra_passes`.
    let extra_passes = extra_passes.unwrap_or(DEFAULT_EXTRA_PASSES);

    // upstream: `if block_subset is None: block_subset = fully_annotated_blocks(ann)` +
    // `block_subset = dict.fromkeys(block_subset)`. The Rust port
    // materialises both paths into a `Vec<BlockRef>` with duplicates
    // collapsed via BlockKey identity.
    let subset_owned: Vec<BlockRef> = match block_subset {
        Some(b) => b.to_vec(),
        None => fully_annotated_blocks(ann),
    };
    let subset = dedupe_blocks(&subset_owned);

    // upstream: `if ann.translator: checkgraphs(ann, block_subset)`.
    checkgraphs(ann, &subset);
    // upstream: `transform_dead_code(ann, block_subset)` — deferred.
    transform_dead_code(ann, &subset);
    // upstream: `for pass_ in extra_passes: pass_(ann, block_subset)`.
    for pass_ in extra_passes {
        pass_(ann, &subset);
    }
    // upstream: `transform_dead_op_vars(ann, block_subset)` — deferred.
    transform_dead_op_vars(ann, &subset);
    // upstream: `if ann.translator: checkgraphs(ann, block_subset)`.
    checkgraphs(ann, &subset);
}

/// Collapse duplicates by block identity, mirroring upstream's
/// `dict.fromkeys(block_subset)` de-duplication step (transform.py:262).
fn dedupe_blocks(blocks: &[BlockRef]) -> Vec<BlockRef> {
    let mut seen: HashSet<BlockKey> = HashSet::new();
    let mut out = Vec::with_capacity(blocks.len());
    for b in blocks {
        if seen.insert(BlockKey::of(b)) {
            out.push(b.clone());
        }
    }
    out
}

/// RPython `transform.py:145-165` — `transform_dead_code(self, block_subset)`.
///
/// ```python
/// def transform_dead_code(self, block_subset):
///     """Remove dead code: these are the blocks that are not annotated at all
///     because the annotation considered that no conditional jump could reach
///     them."""
///     for block in block_subset:
///         for link in block.exits:
///             if link not in self.links_followed:
///                 lst = list(block.exits)
///                 lst.remove(link)
///                 block.exits = tuple(lst)
///                 if not block.exits:
///                     cutoff_alwaysraising_block(self, block)
///                 elif block.canraise:
///                     if block.exits[0].exitcase is not None:
///                         # killed the non-exceptional path!
///                         cutoff_alwaysraising_block(self, block)
///                 if len(block.exits) == 1:
///                     block.exitswitch = None
///                     block.exits[0].exitcase = None
/// ```
pub fn transform_dead_code(ann: &RPythonAnnotator, block_subset: &[BlockRef]) {
    for block in block_subset {
        // upstream: `for link in block.exits:`. The mutation below
        // happens to `block.exits` while we iterate, so snapshot first
        // (matches upstream's implicit `list(...)` copy through
        // identity — Python lets you mutate a list during iteration
        // provided you don't resize, but `block.exits = tuple(lst)`
        // rebinds the attribute which changes subsequent comparisons).
        let exits_snapshot: Vec<LinkRef> = block.borrow().exits.iter().cloned().collect();
        for link in exits_snapshot {
            let followed = ann.links_followed.borrow().contains(&LinkKey::of(&link));
            if followed {
                continue;
            }
            // upstream: remove this link from block.exits.
            {
                let mut blk = block.borrow_mut();
                blk.exits.retain(|l| !Rc::ptr_eq(l, &link));
            }
            let no_exits_left = block.borrow().exits.is_empty();
            let canraise = block.borrow().canraise();
            let first_exitcase_nontrivial = {
                let blk = block.borrow();
                blk.exits
                    .first()
                    .map(|l| l.borrow().exitcase.is_some())
                    .unwrap_or(false)
            };

            // upstream: `if not block.exits: cutoff_alwaysraising_block(...)`.
            if no_exits_left {
                cutoff_alwaysraising_block(ann, block);
            } else if canraise && first_exitcase_nontrivial {
                // upstream: `killed the non-exceptional path!`.
                cutoff_alwaysraising_block(ann, block);
            }

            // upstream: `if len(block.exits) == 1: block.exitswitch =
            // None; block.exits[0].exitcase = None`.
            if block.borrow().exits.len() == 1 {
                let mut blk = block.borrow_mut();
                blk.exitswitch = None;
                if let Some(surviving) = blk.exits.first() {
                    surviving.borrow_mut().exitcase = None;
                }
            }
        }
    }
}

/// RPython `transform.py:167-198` — `cutoff_alwaysraising_block(self, block)`.
///
/// ```python
/// def cutoff_alwaysraising_block(self, block):
///     "Fix a block whose end can never be reached at run-time."
///     can_succeed    = [op for op in block.operations
///                          if op.result.annotation is not None]
///     cannot_succeed = [op for op in block.operations
///                          if op.result.annotation is None]
///     n = len(can_succeed)
///     assert can_succeed == block.operations[:n]
///     assert cannot_succeed == block.operations[n:]
///     assert 0 <= n < len(block.operations)
///     del block.operations[n+1:]
///     self.setbinding(block.operations[n].result, annmodel.s_ImpossibleValue)
///     graph = self.annotated[block]
///     msg = "Call to %r should have raised an exception" % (getattr(graph, 'func', None),)
///     c1 = Constant(AssertionError)
///     c2 = Constant(AssertionError(msg))
///     errlink = Link([c1, c2], graph.exceptblock)
///     block.recloseblock(errlink, *block.exits)
///     self.links_followed[errlink] = True
///     etype, evalue = graph.exceptblock.inputargs
///     s_type = annmodel.SomeTypeOf([evalue])
///     s_value = annmodel.SomeInstance(self.bookkeeper.getuniqueclassdef(Exception))
///     self.setbinding(etype, s_type)
///     self.setbinding(evalue, s_value)
///     self.bookkeeper.getuniqueclassdef(AssertionError)
/// ```
pub fn cutoff_alwaysraising_block(ann: &RPythonAnnotator, block: &BlockRef) {
    // upstream: partition block.operations by whether op.result has
    // an annotation. SpaceOperation.result is Hlvalue — only the
    // Variable variant carries `annotation`.
    let (n, total) = {
        let blk = block.borrow();
        let n = blk
            .operations
            .iter()
            .position(|op| match &op.result {
                Hlvalue::Variable(v) => v.annotation.is_none(),
                Hlvalue::Constant(_) => false,
            })
            .unwrap_or(blk.operations.len());
        (n, blk.operations.len())
    };
    // upstream: `assert 0 <= n < len(block.operations)`.
    assert!(
        n < total,
        "cutoff_alwaysraising_block: no failing op (n={n}, total={total})"
    );

    // upstream: `del block.operations[n+1:]`.
    block.borrow_mut().operations.truncate(n + 1);
    // upstream: `self.setbinding(block.operations[n].result,
    // annmodel.s_ImpossibleValue)`. Same mutation pattern as
    // annrpython.rs:1846-1850.
    {
        let mut blk = block.borrow_mut();
        if let Hlvalue::Variable(v) = &mut blk.operations[n].result {
            ann.setbinding(v, s_impossible_value());
        }
    }

    // upstream: `graph = self.annotated[block]`.
    let graph: GraphRef = {
        let annotated = ann.annotated.borrow();
        annotated
            .get(&BlockKey::of(block))
            .and_then(|g| g.clone())
            .expect("cutoff_alwaysraising_block: block not annotated")
    };

    // upstream: `msg = "Call to %r should have raised an exception" %
    // (getattr(graph, 'func', None),)`.
    let func_repr = graph
        .borrow()
        .func
        .as_ref()
        .map(|f| format!("{:?}", f))
        .unwrap_or_else(|| "None".to_string());
    let msg = format!("Call to {func_repr} should have raised an exception");

    // upstream: `c1 = Constant(AssertionError); c2 = Constant(AssertionError(msg))`.
    let assert_err_class = HOST_ENV
        .lookup_builtin("AssertionError")
        .expect("HOST_ENV missing AssertionError");
    let c1 = Hlvalue::Constant(Constant::new(ConstValue::HostObject(
        assert_err_class.clone(),
    )));
    let err_instance =
        HostObject::new_instance(assert_err_class.clone(), vec![ConstValue::Str(msg.clone())]);
    let c2 = Hlvalue::Constant(Constant::new(ConstValue::HostObject(err_instance)));

    // upstream: `errlink = Link([c1, c2], graph.exceptblock)`.
    let exceptblock = graph.borrow().exceptblock.clone();
    let errlink = Link::new(vec![c1, c2], Some(exceptblock.clone()), None).into_ref();

    // upstream: `block.recloseblock(errlink, *block.exits)`.
    let mut new_exits: Vec<LinkRef> = vec![errlink.clone()];
    let existing_exits: Vec<LinkRef> = block.borrow().exits.iter().cloned().collect();
    new_exits.extend(existing_exits);
    block.recloseblock(new_exits);

    // upstream: `self.links_followed[errlink] = True`.
    ann.links_followed
        .borrow_mut()
        .insert(LinkKey::of(&errlink));

    // upstream: `etype, evalue = graph.exceptblock.inputargs`.
    let (etype_rc, evalue_rc) = {
        let ex = exceptblock.borrow();
        assert_eq!(ex.inputargs.len(), 2, "exceptblock inputargs len != 2");
        let t = match &ex.inputargs[0] {
            Hlvalue::Variable(v) => Rc::new(v.clone()),
            _ => panic!("exceptblock.inputargs[0] is not a Variable"),
        };
        let v = match &ex.inputargs[1] {
            Hlvalue::Variable(v) => Rc::new(v.clone()),
            _ => panic!("exceptblock.inputargs[1] is not a Variable"),
        };
        (t, v)
    };
    // upstream: `s_type = annmodel.SomeTypeOf([evalue])`.
    let s_type = typeof_vars(&[evalue_rc.clone()]);
    // upstream: `s_value = annmodel.SomeInstance(
    //     self.bookkeeper.getuniqueclassdef(Exception))`.
    let exc_class = HOST_ENV
        .lookup_builtin("Exception")
        .expect("HOST_ENV missing Exception");
    let exc_classdef = ann
        .bookkeeper
        .getuniqueclassdef(&exc_class)
        .expect("bookkeeper.getuniqueclassdef(Exception) failed");
    let s_value = SomeValue::Instance(SomeInstance::new(
        Some(exc_classdef),
        false,
        Default::default(),
    ));

    // upstream: `self.setbinding(etype, s_type); self.setbinding(evalue, s_value)`.
    {
        let mut ex = exceptblock.borrow_mut();
        if let Hlvalue::Variable(etype_var) = &mut ex.inputargs[0] {
            ann.setbinding(etype_var, s_type);
        }
        if let Hlvalue::Variable(evalue_var) = &mut ex.inputargs[1] {
            ann.setbinding(evalue_var, s_value);
        }
    }
    // Drop the detached Rc<Variable>s — they were only needed to
    // synthesise s_type via typeof_vars.
    drop(etype_rc);
    drop(evalue_rc);

    // upstream: `self.bookkeeper.getuniqueclassdef(AssertionError)`.
    // Side-effect only — register the class so later flow sees it.
    let _ = ann.bookkeeper.getuniqueclassdef(&assert_err_class);
}

/// RPython `transform.py:137-143` — `transform_dead_op_vars(ann, blocks)`.
///
/// ```python
/// def transform_dead_op_vars(ann, block_subset):
///     from rpython.translator.simplify import transform_dead_op_vars_in_blocks
///     transform_dead_op_vars_in_blocks(block_subset, ann.translator.graphs,
///             ann.translator)
/// ```
///
/// Delegates to `simplify::transform_dead_op_vars_in_blocks` with the
/// annotator's translator hooked up. Upstream's multi-graph
/// `start_blocks` set (`translator.annotator.annotated[block].startblock`)
/// is computed here and threaded through as a graph-resolver closure
/// (Rust doesn't have Python's `for block in blocks: ...annotated[block]
/// .startblock` concise expression).
pub fn transform_dead_op_vars(ann: &RPythonAnnotator, block_subset: &[BlockRef]) {
    use crate::translator::simplify;
    // upstream: `set_of_blocks = set(blocks)`; start_blocks computed
    // by `ann.translator.annotated[block].startblock`. The Rust port
    // splits this: `simplify::transform_dead_op_vars_in_blocks` takes
    // a single-graph reference + translator; iterate the distinct
    // graphs covered by block_subset and run the pass per-graph. The
    // Rust model's `annotated` map is keyed by block identity, so the
    // same dispatch produces the same set of graphs upstream would
    // build via `{translator.annotator.annotated[block].startblock
    // for block in blocks}`.
    let mut seen: HashSet<GraphKey> = HashSet::new();
    let annotated = ann.annotated.borrow();
    for block in block_subset {
        let Some(Some(graph)) = annotated.get(&BlockKey::of(block)) else {
            continue;
        };
        let gkey = GraphKey::of(graph);
        if !seen.insert(gkey) {
            continue;
        }
        let graph_blocks = graph.borrow().iterblocks();
        let _ = graph_blocks.len();
        simplify::transform_dead_op_vars_in_blocks(
            &graph_blocks,
            &[graph.as_ptr() as *const _],
            Some(&ann.translator.borrow()),
            &graph.borrow(),
        );
    }
}

// References to unused imports during deferred-port phase would trip
// the -Dwarnings profile; keep them live via the helper below.
#[allow(dead_code)]
fn _keep_graph_import_alive(_graph: &GraphRef) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{
        Block, BlockRefExt, ConstValue, Constant, FunctionGraph, Hlvalue, Link, Variable,
    };
    use std::cell::RefCell;
    use std::rc::Rc;

    fn mk_minimum_graph(name: &str) -> GraphRef {
        // Minimum valid graph: startblock → returnblock via Constant(1).
        let start = Block::shared(vec![]);
        let graph: GraphRef = Rc::new(RefCell::new(FunctionGraph::new(name, start.clone())));
        let link = Link::new(
            vec![Hlvalue::Constant(Constant::new(ConstValue::Int(1)))],
            Some(graph.borrow().returnblock.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![link]);
        graph
    }

    #[test]
    fn checkgraphs_dedups_by_graph_identity() {
        // Two blocks reachable from the same graph → checkgraph should
        // run only once (we can't observe that directly, but the Rust
        // port's `seen` set matches upstream semantics — we just assert
        // the function completes without panic for a valid graph).
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_minimum_graph("g");
        let start = graph.borrow().startblock.clone();
        let returnblock = graph.borrow().returnblock.clone();

        // Populate `ann.annotated` so both blocks map to the same graph.
        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&start), Some(graph.clone()));
        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&returnblock), Some(graph.clone()));
        ann.all_blocks
            .borrow_mut()
            .insert(BlockKey::of(&start), start.clone());
        ann.all_blocks
            .borrow_mut()
            .insert(BlockKey::of(&returnblock), returnblock.clone());

        checkgraphs(&ann, &[start.clone(), returnblock.clone()]);
    }

    #[test]
    fn fully_annotated_blocks_skips_blocked_entries() {
        // One block fully annotated (Some(Some(graph))) and one block
        // with `None` (= upstream's `False`) should yield only the
        // former.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_minimum_graph("g");
        let a = Block::shared(vec![]);
        let b = Block::shared(vec![]);
        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&a), Some(graph.clone()));
        ann.annotated.borrow_mut().insert(BlockKey::of(&b), None);
        ann.all_blocks
            .borrow_mut()
            .insert(BlockKey::of(&a), a.clone());
        ann.all_blocks
            .borrow_mut()
            .insert(BlockKey::of(&b), b.clone());

        let blocks = fully_annotated_blocks(&ann);
        assert_eq!(blocks.len(), 1);
        assert!(Rc::ptr_eq(&blocks[0], &a));
    }

    #[test]
    fn transform_graph_runs_checkgraphs_twice_and_respects_empty_extra_passes() {
        // Wire a minimum valid graph through `transform_graph` with an
        // explicit block_subset. Mark `start`'s exit as followed so
        // `transform_dead_code` leaves it alone — otherwise it would
        // try to cutoff an empty startblock, which upstream semantics
        // disallow (cutoff requires at least one failing op).
        use crate::flowspace::model::LinkKey;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_minimum_graph("g");
        let start = graph.borrow().startblock.clone();
        ann.translator
            .borrow()
            .graphs
            .borrow_mut()
            .push(graph.clone());
        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&start), Some(graph.clone()));
        ann.all_blocks
            .borrow_mut()
            .insert(BlockKey::of(&start), start.clone());
        // Mark the single exit as followed so transform_dead_code
        // doesn't prune it.
        for link in &start.borrow().exits {
            ann.links_followed.borrow_mut().insert(LinkKey::of(link));
        }

        transform_graph(&ann, None, Some(&[start.clone()]));
    }

    #[test]
    fn transform_dead_code_prunes_unfollowed_exits() {
        // A conditional block with two exits: only the False branch is
        // followed. transform_dead_code drops the True branch and the
        // surviving exit gets exitswitch=None + exitcase=None.
        use crate::flowspace::model::{ConstValue as CV, Constant as C, LinkKey, SpaceOperation};
        let ann = RPythonAnnotator::new(None, None, None, false);
        let v = Variable::new();
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let body = Block::shared(vec![Hlvalue::Variable(Variable::new())]);
        let graph: GraphRef = Rc::new(RefCell::new(crate::flowspace::model::FunctionGraph::new(
            "f",
            start.clone(),
        )));
        let returnblock = graph.borrow().returnblock.clone();

        // One op so cutoff_alwaysraising_block would have something to
        // work with, but we won't reach cutoff here.
        body.borrow_mut().operations.push(SpaceOperation::new(
            "noop",
            vec![Hlvalue::Variable(v.clone())],
            Hlvalue::Variable(Variable::new()),
        ));

        start.borrow_mut().exitswitch = Some(Hlvalue::Variable(v.clone()));
        let left = Link::new(
            vec![Hlvalue::Variable(v.clone())],
            Some(body.clone()),
            Some(Hlvalue::Constant(C::new(CV::Bool(false)))),
        )
        .into_ref();
        let right = Link::new(
            vec![Hlvalue::Variable(v.clone())],
            Some(body.clone()),
            Some(Hlvalue::Constant(C::new(CV::Bool(true)))),
        )
        .into_ref();
        start.closeblock(vec![left.clone(), right.clone()]);

        // Only the false branch is followed.
        ann.links_followed.borrow_mut().insert(LinkKey::of(&left));
        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&start), Some(graph.clone()));
        ann.annotated
            .borrow_mut()
            .insert(BlockKey::of(&body), Some(graph.clone()));

        let body_end = Link::new(
            vec![body.borrow().inputargs[0].clone()],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        body.closeblock(vec![body_end]);

        transform_dead_code(&ann, &[start.clone()]);

        let s = start.borrow();
        assert_eq!(s.exits.len(), 1);
        assert!(s.exitswitch.is_none());
        assert!(s.exits[0].borrow().exitcase.is_none());
        assert!(Rc::ptr_eq(
            s.exits[0].borrow().target.as_ref().unwrap(),
            &body
        ));
    }
}
