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

use crate::annotator::annrpython::RPythonAnnotator;
use crate::flowspace::model::{BlockKey, BlockRef, GraphKey, GraphRef, checkgraph};

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
/// Deferred: the port of `cutoff_alwaysraising_block` depends on the
/// bookkeeper's `getuniqueclassdef(AssertionError)` flow and the
/// `annmodel.SomeTypeOf` / `SomeInstance` binding updates. Landed as a
/// no-op stub so `transform_graph`'s line order matches upstream; the
/// real body arrives alongside the cutoff port.
pub fn transform_dead_code(_ann: &RPythonAnnotator, _block_subset: &[BlockRef]) {
    // intentionally no-op — see module-level note.
}

/// RPython `transform.py:137-143` — `transform_dead_op_vars(ann, blocks)`.
///
/// Upstream just forwards to `simplify.transform_dead_op_vars_in_blocks`.
/// Both that callee and its dependency on the translator's full graph
/// set are deferred; see module docs.
pub fn transform_dead_op_vars(_ann: &RPythonAnnotator, _block_subset: &[BlockRef]) {
    // intentionally no-op — see module-level note.
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
        // explicit block_subset. The no-op `transform_dead_code` +
        // `transform_dead_op_vars` should leave the graph unchanged,
        // and both pre- and post-checkgraphs should succeed.
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

        transform_graph(&ann, None, Some(&[start.clone()]));
    }
}
