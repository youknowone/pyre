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
//! Port coverage:
//!   - `checkgraphs` (transform.py:13-19)
//!   - `fully_annotated_blocks` (transform.py:21-25)
//!   - `transform_allocate` (transform.py:36-50)
//!   - `transform_extend_with_str_slice` (transform.py:59-75)
//!   - `transform_extend_with_char_count` (transform.py:84-106)
//!   - `transform_list_contains` (transform.py:115-134)
//!   - `transform_dead_op_vars` (transform.py:137-143) — forwards to
//!      [`crate::translator::simplify::transform_dead_op_vars_in_blocks`].
//!   - `transform_dead_code` + `cutoff_alwaysraising_block`
//!      (transform.py:145-198).
//!   - `default_extra_passes` + `transform_graph` (transform.py:246-272).
//!
//! `insert_ll_stackcheck` (transform.py:200-243) is rtyper-phase and
//! therefore out of scope for the annotator-phase port.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::annotator::annrpython::RPythonAnnotator;
use crate::annotator::model::{
    KnownType, SomeInstance, SomeObjectTrait, SomeValue, SomeValueTag, s_impossible_value,
    typeof_vars,
};
use crate::flowspace::model::{
    BlockKey, BlockRef, BlockRefExt, ConstValue, Constant, GraphKey, GraphRef, HOST_ENV, Hlvalue,
    HostObject, Link, LinkKey, LinkRef, SpaceOperation, Variable, checkgraph,
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
/// ```python
/// default_extra_passes = [
///     transform_allocate,
///     transform_extend_with_str_slice,
///     transform_extend_with_char_count,
///     transform_list_contains,
/// ]
/// ```
///
/// The four pattern-rewriting passes are defined in this module and
/// stay available as freestanding functions, but three of them
/// (`transform_allocate`, `transform_extend_with_str_slice`,
/// `transform_extend_with_char_count`) are temporarily held back
/// from this list until their synthetic opnames
/// (`alloc_and_set`, `extend_with_str_slice`, `extend_with_char_count`)
/// are registered in [`crate::flowspace::operation::OpKind`]. Upstream
/// never reflows blocks after `transform_graph` fires, so the
/// annotator never has to look up these opnames in `OpKind`; the Rust
/// port can't rely on that invariant yet because
/// [`RPythonAnnotator::reflowpendingblock`] may be reached from later
/// helper-graph specialisation paths and `flowin_op_loop` panics on
/// unknown raising opnames. `transform_list_contains` is currently a
/// no-op (see its own doc comment) so it's safe to keep wired.
pub const DEFAULT_EXTRA_PASSES: &[TransformPass] = &[transform_list_contains];

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
/// annotator's translator hooked up. Upstream passes the block subset
/// directly — `block_subset` may be smaller than any one graph's full
/// block list (e.g. `complete_helpers()` threads `self.added_blocks`
/// only). The pass mutates only those blocks; `dependencies` /
/// `read_vars` still flow correctly because cross-subset link targets
/// contribute `read_vars` for their inputargs via the `target not in
/// set_of_blocks` branch.
///
/// Upstream's `start_blocks` set (`{translator.annotator.annotated[block]
/// .startblock for block in blocks}`) is reproduced per-graph: the
/// Rust port iterates the distinct owning graphs covered by
/// `block_subset`, grouping the subset by graph and forwarding only
/// that subset to `transform_dead_op_vars_in_blocks`. The
/// `single_graph` argument names the per-graph startblock whose
/// inputargs must survive.
pub fn transform_dead_op_vars(ann: &RPythonAnnotator, block_subset: &[BlockRef]) {
    use crate::translator::simplify;
    let annotated = ann.annotated.borrow();
    // upstream's `start_blocks = {...annotated[block].startblock for
    // block in blocks}` — group the requested blocks by their owning
    // graph, dropping unannotated ones (upstream would raise KeyError
    // here; the Rust port follows the `if block in annotated` guard
    // the callers already observe).
    let mut groups: HashMap<GraphKey, (GraphRef, Vec<BlockRef>)> = HashMap::new();
    for block in block_subset {
        let Some(Some(graph)) = annotated.get(&BlockKey::of(block)) else {
            continue;
        };
        let entry = groups
            .entry(GraphKey::of(graph))
            .or_insert_with(|| (graph.clone(), Vec::new()));
        entry.1.push(block.clone());
    }
    drop(annotated);

    for (_gkey, (graph, subset)) in groups {
        simplify::transform_dead_op_vars_in_blocks(
            &subset,
            &[graph.as_ptr() as *const _],
            Some(&ann.translator.borrow()),
            &graph.borrow(),
        );
    }
}

/// RPython `transform.py:36-50` — `transform_allocate(self, block_subset)`.
///
/// ```python
/// def transform_allocate(self, block_subset):
///     """Transforms [a] * b to alloc_and_set(b, a) where b is int."""
///     for block in block_subset:
///         length1_lists = {}   # maps 'c' to 'a', in the above notation
///         for i in range(len(block.operations)):
///             op = block.operations[i]
///             if (op.opname == 'newlist' and
///                 len(op.args) == 1):
///                 length1_lists[op.result] = op.args[0]
///             elif (op.opname == 'mul' and
///                   op.args[0] in length1_lists):
///                 new_op = SpaceOperation('alloc_and_set',
///                                         (op.args[1], length1_lists[op.args[0]]),
///                                         op.result)
///                 block.operations[i] = new_op
/// ```
pub fn transform_allocate(_ann: &RPythonAnnotator, block_subset: &[BlockRef]) {
    for block in block_subset {
        // upstream: maps `c` (newlist result) → `a` (newlist's only arg).
        let mut length1_lists: HashMap<Variable, Hlvalue> = HashMap::new();
        let ops_len = block.borrow().operations.len();
        for i in 0..ops_len {
            // upstream: `op = block.operations[i]`.
            let op = block.borrow().operations[i].clone();
            if op.opname == "newlist" && op.args.len() == 1 {
                if let Hlvalue::Variable(result_v) = &op.result {
                    length1_lists.insert(result_v.clone(), op.args[0].clone());
                }
            } else if op.opname == "mul" {
                // upstream: `op.args[0] in length1_lists`.
                let Some(Hlvalue::Variable(first_var)) = op.args.first() else {
                    continue;
                };
                let Some(a_value) = length1_lists.get(first_var) else {
                    continue;
                };
                let Some(b_value) = op.args.get(1) else {
                    continue;
                };
                // upstream: `SpaceOperation('alloc_and_set', (op.args[1], length1_lists[op.args[0]]), op.result)`.
                let new_op = SpaceOperation::with_offset(
                    "alloc_and_set",
                    vec![b_value.clone(), a_value.clone()],
                    op.result.clone(),
                    op.offset,
                );
                block.borrow_mut().operations[i] = new_op;
            }
        }
    }
}

/// RPython `transform.py:59-75` — `transform_extend_with_str_slice(self, block_subset)`.
///
/// ```python
/// def transform_extend_with_str_slice(self, block_subset):
///     """Transforms lst += string[x:y] to extend_with_str_slice"""
///     for block in block_subset:
///         slice_sources = {}
///         for i in range(len(block.operations)):
///             op = block.operations[i]
///             if (op.opname == 'getslice' and
///                 self.gettype(op.args[0]) is str):
///                 slice_sources[op.result] = op.args
///             elif (op.opname == 'inplace_add' and
///                   op.args[1] in slice_sources and
///                   self.gettype(op.args[0]) is list):
///                 v_string, v_x, v_y = slice_sources[op.args[1]]
///                 new_op = SpaceOperation('extend_with_str_slice',
///                                         [op.args[0], v_x, v_y, v_string],
///                                         op.result)
///                 block.operations[i] = new_op
/// ```
///
/// `self.gettype(v) is str` routes through `ann.gettype(v) ==
/// KnownType::Str`, which — like upstream — matches both `SomeString`
/// and `SomeChar` because both set `knowntype = str`. The equivalent
/// `is list` check goes through `SomeValue::List(_)` since the Rust
/// port's [`KnownType`] enum collapses SomeList to [`KnownType::Other`]
/// (list variant pending).
pub fn transform_extend_with_str_slice(ann: &RPythonAnnotator, block_subset: &[BlockRef]) {
    for block in block_subset {
        // upstream: maps `b` (getslice result Variable) → [string, x, y].
        let mut slice_sources: HashMap<Variable, Vec<Hlvalue>> = HashMap::new();
        let ops_len = block.borrow().operations.len();
        for i in 0..ops_len {
            let op = block.borrow().operations[i].clone();
            if op.opname == "getslice" && op.args.len() >= 3 {
                // upstream: `self.gettype(op.args[0]) is str` — true
                // for both SomeString and SomeChar (both set
                // knowntype = str).
                if ann.gettype(&op.args[0]) == KnownType::Str {
                    if let Hlvalue::Variable(result_v) = &op.result {
                        slice_sources.insert(result_v.clone(), op.args.clone());
                    }
                }
            } else if op.opname == "inplace_add" && op.args.len() == 2 {
                // upstream: `op.args[1] in slice_sources`.
                let Hlvalue::Variable(arg1_v) = &op.args[1] else {
                    continue;
                };
                let Some(source_args) = slice_sources.get(arg1_v) else {
                    continue;
                };
                // upstream: `self.gettype(op.args[0]) is list`.
                if !matches!(ann.annotation(&op.args[0]), Some(SomeValue::List(_))) {
                    continue;
                }
                // upstream: `v_string, v_x, v_y = slice_sources[op.args[1]]`.
                let v_string = source_args[0].clone();
                let v_x = source_args[1].clone();
                let v_y = source_args[2].clone();
                let new_op = SpaceOperation::with_offset(
                    "extend_with_str_slice",
                    vec![op.args[0].clone(), v_x, v_y, v_string],
                    op.result.clone(),
                    op.offset,
                );
                block.borrow_mut().operations[i] = new_op;
            }
        }
    }
}

/// RPython `transform.py:84-106` — `transform_extend_with_char_count(self, block_subset)`.
///
/// ```python
/// def transform_extend_with_char_count(self, block_subset):
///     """Transforms lst += char*count to extend_with_char_count"""
///     for block in block_subset:
///         mul_sources = {}
///         for i in range(len(block.operations)):
///             op = block.operations[i]
///             if op.opname == 'mul':
///                 s0 = self.annotation(op.args[0])
///                 s1 = self.annotation(op.args[1])
///                 if (isinstance(s0, annmodel.SomeChar) and
///                     isinstance(s1, annmodel.SomeInteger)):
///                     mul_sources[op.result] = op.args[0], op.args[1]
///                 elif (isinstance(s1, annmodel.SomeChar) and
///                       isinstance(s0, annmodel.SomeInteger)):
///                     mul_sources[op.result] = op.args[1], op.args[0]
///             elif (op.opname == 'inplace_add' and
///                   op.args[1] in mul_sources and
///                   self.gettype(op.args[0]) is list):
///                 v_char, v_count = mul_sources[op.args[1]]
///                 new_op = SpaceOperation('extend_with_char_count',
///                                         [op.args[0], v_char, v_count],
///                                         op.result)
///                 block.operations[i] = new_op
/// ```
pub fn transform_extend_with_char_count(ann: &RPythonAnnotator, block_subset: &[BlockRef]) {
    // upstream: `isinstance(s, annmodel.SomeInteger)` — Python's
    // `isinstance` walks the MRO, so `SomeBool` (which subclasses
    // `SomeInteger`) matches too. Reproduce that via
    // [`SomeValueTag::mro`] so `SomeBool` / `SomeInteger` both
    // qualify as integer counts.
    fn is_integer_subclass(s: &Option<SomeValue>) -> bool {
        s.as_ref()
            .is_some_and(|sv| sv.tag().mro().iter().any(|t| *t == SomeValueTag::Integer))
    }

    for block in block_subset {
        // upstream: maps `b` (mul result Variable) → (char, count).
        let mut mul_sources: HashMap<Variable, (Hlvalue, Hlvalue)> = HashMap::new();
        let ops_len = block.borrow().operations.len();
        for i in 0..ops_len {
            let op = block.borrow().operations[i].clone();
            if op.opname == "mul" && op.args.len() == 2 {
                // upstream: `s0 = self.annotation(op.args[0])`.
                let s0 = ann.annotation(&op.args[0]);
                let s1 = ann.annotation(&op.args[1]);
                let Hlvalue::Variable(result_v) = &op.result else {
                    continue;
                };
                // upstream: SomeChar * SomeInteger.
                if matches!(s0, Some(SomeValue::Char(_))) && is_integer_subclass(&s1) {
                    mul_sources.insert(result_v.clone(), (op.args[0].clone(), op.args[1].clone()));
                } else if matches!(s1, Some(SomeValue::Char(_))) && is_integer_subclass(&s0) {
                    // upstream: swap so char is first.
                    mul_sources.insert(result_v.clone(), (op.args[1].clone(), op.args[0].clone()));
                }
            } else if op.opname == "inplace_add" && op.args.len() == 2 {
                let Hlvalue::Variable(arg1_v) = &op.args[1] else {
                    continue;
                };
                let Some((v_char, v_count)) = mul_sources.get(arg1_v).cloned() else {
                    continue;
                };
                if !matches!(ann.annotation(&op.args[0]), Some(SomeValue::List(_))) {
                    continue;
                }
                let new_op = SpaceOperation::with_offset(
                    "extend_with_char_count",
                    vec![op.args[0].clone(), v_char, v_count],
                    op.result.clone(),
                    op.offset,
                );
                block.borrow_mut().operations[i] = new_op;
            }
        }
    }
}

/// RPython `transform.py:115-134` — `transform_list_contains(self, block_subset)`.
///
/// ```python
/// def transform_list_contains(self, block_subset):
///     """Transforms x in [2, 3]"""
///     for block in block_subset:
///         newlist_sources = {}
///         for i in range(len(block.operations)):
///             op = block.operations[i]
///             if op.opname == 'newlist':
///                 newlist_sources[op.result] = op.args
///             elif op.opname == 'contains' and op.args[0] in newlist_sources:
///                 items = {}
///                 for v in newlist_sources[op.args[0]]:
///                     s = self.annotation(v)
///                     if not s.is_immutable_constant():
///                         break
///                     items[s.const] = None
///                 else:
///                     # all arguments of the newlist are annotation constants
///                     op.args[0] = Constant(items)
///                     s_dict = self.annotation(op.args[0])
///                     s_dict.dictdef.generalize_key(self.binding(op.args[1]))
/// ```
///
/// The port runs as an intentional no-op pending the rtyper-phase
/// port. Two Rust-side prerequisites block the rewrite:
///
/// 1. **ConstValue key typing.** Upstream stores `s.const` (any
///    immutable Python value — int, str, float …) as the dict key,
///    but the Rust [`ConstValue::Dict`] variant is
///    `HashMap<String, ConstValue>` and only supports string keys. A
///    faithful rewrite of `[2, 3]` into a `Dict{ ... }` cannot
///    preserve the original keys, and any string-substitute scheme
///    (e.g. `format!("{:?}", key)`) changes the runtime dict to be
///    keyed by debug-formatted strings — `contains(2, dict)` would
///    then return False and the program would silently miscompile.
/// 2. **Downstream lowering.** The rewrite's sole benefit is runtime
///    O(1) dict lookup in place of linear list scan, which only
///    materialises once the rtyper lowers the Constant(dict) into a
///    real hash-map literal. The Rust port is currently
///    annotator-only; no rtyper exists to consume the rewritten
///    `contains(x, dict)`. Rewriting without a consumer has no
///    observable effect (since the rewritten Constant never
///    executes) while risking the miscompilation path above.
///
/// The pass therefore stays in [`DEFAULT_EXTRA_PASSES`] for
/// line-by-line parity with upstream's pass list, but with an empty
/// body until the rtyper port lands and [`ConstValue::Dict`] (or a
/// new companion variant) supports arbitrary-typed keys.
pub fn transform_list_contains(_ann: &RPythonAnnotator, _block_subset: &[BlockRef]) {
    // Intentional no-op — see the doc comment above.
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

    #[test]
    fn transform_allocate_rewrites_newlist_plus_mul_to_alloc_and_set() {
        // Port of upstream's shape:
        //   c = newlist(a)
        //   d = mul(c, b)
        //   -->
        //   d = alloc_and_set(b, a)
        use crate::flowspace::model::{ConstValue as CV, Constant as C, SpaceOperation};
        let ann = RPythonAnnotator::new(None, None, None, false);
        let block = Block::shared(vec![]);
        let a = Variable::new();
        let b = Variable::new();
        let c = Variable::new();
        let d = Variable::new();
        // newlist(a) → c
        block.borrow_mut().operations.push(SpaceOperation::new(
            "newlist",
            vec![Hlvalue::Variable(a.clone())],
            Hlvalue::Variable(c.clone()),
        ));
        // mul(c, b) → d
        block.borrow_mut().operations.push(SpaceOperation::new(
            "mul",
            vec![Hlvalue::Variable(c.clone()), Hlvalue::Variable(b.clone())],
            Hlvalue::Variable(d.clone()),
        ));
        // Unrelated op to prove the pass doesn't touch foreign muls.
        block.borrow_mut().operations.push(SpaceOperation::new(
            "mul",
            vec![
                Hlvalue::Variable(Variable::new()),
                Hlvalue::Constant(C::new(CV::Int(2))),
            ],
            Hlvalue::Variable(Variable::new()),
        ));

        transform_allocate(&ann, &[block.clone()]);

        let ops = &block.borrow().operations;
        assert_eq!(ops[0].opname, "newlist");
        assert_eq!(ops[1].opname, "alloc_and_set");
        // upstream: args become (b, a).
        let Hlvalue::Variable(arg0) = &ops[1].args[0] else {
            panic!("alloc_and_set arg0 should be Variable");
        };
        assert_eq!(arg0, &b);
        let Hlvalue::Variable(arg1) = &ops[1].args[1] else {
            panic!("alloc_and_set arg1 should be Variable");
        };
        assert_eq!(arg1, &a);
        // Unrelated mul is left alone.
        assert_eq!(ops[2].opname, "mul");
    }

    #[test]
    fn transform_extend_with_str_slice_rewrites_when_types_match() {
        use crate::annotator::listdef::ListDef;
        use crate::annotator::model::{SomeList, SomeString, SomeValue};
        use crate::flowspace::model::SpaceOperation;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let block = Block::shared(vec![]);

        let mut s = Variable::new(); // string source
        let x = Variable::new();
        let y = Variable::new();
        let mut lst = Variable::new(); // list target
        let sliced = Variable::new(); // getslice result
        let result = Variable::new();

        // Pre-bind annotations so the pass's type checks succeed.
        s.annotation = Some(std::rc::Rc::new(SomeValue::String(SomeString::new(
            false, false,
        ))));
        lst.annotation = Some(std::rc::Rc::new(SomeValue::List(SomeList::new(
            ListDef::new(None, SomeValue::Impossible, false, false),
        ))));

        // getslice(s, x, y) → sliced
        block.borrow_mut().operations.push(SpaceOperation::new(
            "getslice",
            vec![
                Hlvalue::Variable(s.clone()),
                Hlvalue::Variable(x.clone()),
                Hlvalue::Variable(y.clone()),
            ],
            Hlvalue::Variable(sliced.clone()),
        ));
        // inplace_add(lst, sliced) → result
        block.borrow_mut().operations.push(SpaceOperation::new(
            "inplace_add",
            vec![
                Hlvalue::Variable(lst.clone()),
                Hlvalue::Variable(sliced.clone()),
            ],
            Hlvalue::Variable(result.clone()),
        ));

        transform_extend_with_str_slice(&ann, &[block.clone()]);

        let ops = &block.borrow().operations;
        assert_eq!(ops[0].opname, "getslice");
        assert_eq!(ops[1].opname, "extend_with_str_slice");
        // upstream: args become [lst, x, y, string].
        let Hlvalue::Variable(arg0) = &ops[1].args[0] else {
            panic!();
        };
        assert_eq!(arg0, &lst);
        let Hlvalue::Variable(arg1) = &ops[1].args[1] else {
            panic!();
        };
        assert_eq!(arg1, &x);
        let Hlvalue::Variable(arg2) = &ops[1].args[2] else {
            panic!();
        };
        assert_eq!(arg2, &y);
        let Hlvalue::Variable(arg3) = &ops[1].args[3] else {
            panic!();
        };
        assert_eq!(arg3, &s);
    }
}
