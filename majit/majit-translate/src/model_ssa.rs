//! `crate::model` (codewriter graph) port of RPython
//! `rpython/translator/backendopt/ssa.py` `SSA_to_SSI` plus the
//! `DataFlowFamilyBuilder.complete` family computation it relies on.
//!
//! The flowspace-model port of the same algorithm lives in
//! [`crate::translator::backendopt::ssa`].  The codewriter pipeline,
//! however, operates on the separate index-based
//! [`crate::model::FunctionGraph`] — Spine A's `transform_graph_to_jitcode`
//! receives one directly and Spine B's `jtransform_opname::lower_graph`
//! produces one — so the same conversion is re-expressed here against
//! that representation.  Both share the underlying
//! [`crate::flowspace::model::Variable`] identity (`id`-keyed `Eq`/`Hash`),
//! so the `UnionFind` families behave identically.
//!
//! ```python
//! def SSA_to_SSI(graph, annotator=None):
//!     entrymap = mkentrymap(graph)
//!     del entrymap[graph.startblock]
//!     variable_families = DataFlowFamilyBuilder(graph).get_variable_families()
//!     pending = []
//!     for block in graph.iterblocks():
//!         if block not in entrymap: continue
//!         variables_created = variables_created_in(block)
//!         seen = set(variables_created)
//!         variables_used = [used-but-not-created vars, in order]
//!         for v in variables_used:
//!             if isinstance(v, Variable) and v._name not in
//!                     ('last_exception_', 'last_exc_value_'):
//!                 pending.append((block, v))
//!     while pending:
//!         block, v = pending.pop()
//!         v_rep = variable_families.find_rep(v)
//!         if v in variables_created_in(block): continue
//!         for w in variables_created:
//!             if variable_families.find_rep(w) is v_rep:
//!                 block.renamevariables({v: w}); break
//!         else:
//!             w = v.copy()
//!             variable_families.union(v, w)
//!             block.renamevariables({v: w})
//!             block.inputargs.append(w)
//!             for link in entrymap[block]:
//!                 link.args.append(v); pending.append((link.prevblock, v))
//! ```
//!
//! Every iteration that touches a `HashMap`/`HashSet` is ordered through
//! [`FunctionGraph::iterblocks_order`] (block-id order) or a `Vec` so the
//! pass is deterministic — the build-time codegen cache keys on output
//! content, and non-deterministic `union` order would change the family
//! representatives the reuse search picks.

use std::collections::{HashMap, HashSet};

use crate::flowspace::model::Variable;
use crate::model::{BlockId, ExitSwitch, FunctionGraph, LinkArg, SpaceOperation};
use crate::tool::algo::unionfind::UnionFind;

/// Incoming-link index: `target -> [(prevblock, exit index in prevblock.exits)]`.
///
/// RPython `mkentrymap` returns `{block: [Link, ...]}`; pyre's `Link`s
/// are owned inside `prevblock.exits`, so an incoming link is addressed
/// by its `(prevblock, index)` pair, which stays valid for the lifetime
/// of the pass (`SSA_to_SSI` only appends to `inputargs` / `link.args`,
/// never reorders blocks or exits).
type EntryMap = HashMap<BlockId, Vec<(BlockId, usize)>>;

/// RPython `mkentrymap(graph)` then `del entrymap[graph.startblock]`
/// (`ssa.py:141-142`).  Scans every block's exits; the startblock has no
/// real incoming edge, so removing it matches upstream's delete.
fn build_entrymap(graph: &FunctionGraph) -> EntryMap {
    let mut map: EntryMap = HashMap::new();
    for src in graph.iterblocks_order() {
        let exits_len = graph.block(src).exits.len();
        for idx in 0..exits_len {
            let target = graph.block(src).exits[idx].target;
            map.entry(target).or_default().push((src, idx));
        }
    }
    map.remove(&graph.startblock);
    map
}

/// One `DataFlowFamilyBuilder` unification opportunity: a block input
/// variable grouped with the matching nth arg of each incoming link.
struct Opportunity {
    inputvar: Variable,
    linkvars: Vec<Variable>,
}

/// RPython `DataFlowFamilyBuilder(graph).get_variable_families()`
/// (`ssa.py:4-90`), restricted to the `complete()` pass (`SSA_to_SSI`
/// never needs `merge_identical_phi_nodes`).
///
/// Opportunities with any `Constant` link arg are dropped — upstream
/// routes them to `opportunities_with_const`, which `complete()` skips.
fn compute_variable_families(
    graph: &FunctionGraph,
    entrymap: &EntryMap,
) -> UnionFind<Variable, ()> {
    let mut opportunities: Vec<Opportunity> = Vec::new();
    // Iterate targets in block-id order (deterministic) rather than over
    // `entrymap`'s `HashMap` order.
    for target in graph.iterblocks_order() {
        let Some(links) = entrymap.get(&target) else {
            continue;
        };
        let inputargs = graph.block(target).inputargs.clone();
        for (n, inputvar) in inputargs.iter().enumerate() {
            let mut linkvars: Vec<Variable> = Vec::with_capacity(links.len());
            let mut has_const = false;
            for &(src, idx) in links {
                match graph.block(src).exits[idx].args.get(n) {
                    Some(LinkArg::Value(v)) => linkvars.push(v.clone()),
                    // Constant arg → upstream `opportunities_with_const`;
                    // arity mismatch (a global-SSA link with fewer args
                    // than the target has inputargs) likewise can't form
                    // a Variable opportunity.
                    Some(LinkArg::Const(_)) | None => has_const = true,
                }
            }
            if !has_const && linkvars.len() == links.len() {
                opportunities.push(Opportunity {
                    inputvar: inputvar.clone(),
                    linkvars,
                });
            }
        }
    }

    let mut families: UnionFind<Variable, ()> = UnionFind::new(|_: &Variable| ());
    // RPython `complete()`: union to a fixpoint.
    let mut progress = true;
    while progress {
        progress = false;
        let mut pending: Vec<Opportunity> = Vec::new();
        for opp in std::mem::take(&mut opportunities) {
            let mut repvars: Vec<Variable> = Vec::with_capacity(1 + opp.linkvars.len());
            repvars.push(families.find_rep(opp.inputvar.clone()));
            for v in &opp.linkvars {
                repvars.push(families.find_rep(v.clone()));
            }
            // Insertion-ordered unique reps (RPython `dict.fromkeys`).
            let mut unique: Vec<Variable> = Vec::new();
            let mut seen: HashSet<Variable> = HashSet::new();
            for v in &repvars {
                if seen.insert(v.clone()) {
                    unique.push(v.clone());
                }
            }
            match unique.len() {
                n if n > 2 => {
                    // Recycle with representatives (RPython
                    // `pending_opportunities.append(vars[:1] + repvars)`).
                    let inputvar = repvars[0].clone();
                    let linkvars = repvars[1..].to_vec();
                    pending.push(Opportunity { inputvar, linkvars });
                }
                2 => {
                    families.union(unique[0].clone(), unique[1].clone());
                    progress = true;
                }
                _ => {}
            }
        }
        opportunities = pending;
    }
    families
}

/// RPython `variables_created_in(block)` (`ssa.py:128-132`): inputargs
/// plus every op result.
fn variables_created_in(graph: &FunctionGraph, id: BlockId) -> HashSet<Variable> {
    let b = graph.block(id);
    let mut s: HashSet<Variable> = HashSet::with_capacity(b.inputargs.len() + b.operations.len());
    for v in &b.inputargs {
        s.insert(v.clone());
    }
    for op in &b.operations {
        if let Some(r) = &op.result {
            s.insert(r.clone());
        }
    }
    s
}

/// RPython `Block.renamevariables({v: w})` (`flowspace/model.py:238-244`)
/// specialised to a single `v -> w` rewrite: rewrites every variable
/// occurrence in `id`'s inputargs, op args/results, exitswitch, and exit
/// link args.  `v` is a used-but-not-created variable, so it appears only
/// in uses; rewriting the definition slots too is a harmless no-op that
/// keeps the call line-for-line with upstream's "rename everywhere".
fn renamevariables(graph: &mut FunctionGraph, id: BlockId, v: &Variable, w: &Variable) {
    let remap = |x: &Variable| -> Variable { if x == v { w.clone() } else { x.clone() } };
    let (new_inputargs, new_ops, new_switch, new_exits) = {
        let b = graph.block(id);
        let new_inputargs: Vec<Variable> = b.inputargs.iter().map(|x| remap(x)).collect();
        let new_ops: Vec<SpaceOperation> = b
            .operations
            .iter()
            .map(|op| SpaceOperation {
                result: op.result.as_ref().map(|r| remap(r)),
                kind: crate::inline::remap_op_kind(&op.kind, &remap),
            })
            .collect();
        let (switch, exits) =
            crate::model::remap_control_flow_metadata_var(&b.exitswitch, &b.exits, &remap, |blk| {
                blk
            });
        (new_inputargs, new_ops, switch, exits)
    };
    let bm = graph.block_mut(id);
    bm.inputargs = new_inputargs;
    bm.operations = new_ops;
    bm.exitswitch = new_switch;
    bm.exits = new_exits;
}

/// RPython `SSA_to_SSI(graph)` (`ssa.py:135-196`) against the codewriter
/// model graph.  Threads every used-but-not-created variable up its
/// incoming links until it reaches a block that already defines a value
/// in the same family, restoring the SSI invariant the register
/// allocator (`regalloc.rs`) assumes.  Idempotent on graphs already in
/// SSI form: their per-block used-set is a subset of the created-set, so
/// `pending` starts empty.
pub fn ssa_to_ssi(graph: &mut FunctionGraph) {
    let entrymap = build_entrymap(graph);
    let mut families = compute_variable_families(graph, &entrymap);

    // Initial pending list (ssa.py:147-169): used-but-not-created vars
    // per non-start block, in block-id then in-block order.
    let mut pending: Vec<(BlockId, Variable)> = Vec::new();
    for id in graph.iterblocks_order() {
        if !entrymap.contains_key(&id) {
            continue;
        }
        let created = variables_created_in(graph, id);
        let mut seen = created.clone();
        let mut used: Vec<Variable> = Vec::new();
        {
            let b = graph.block(id);
            for op in &b.operations {
                for arg in crate::inline::op_variable_refs(&op.kind) {
                    if seen.insert(arg.clone()) {
                        used.push(arg);
                    }
                }
            }
            match &b.exitswitch {
                Some(ExitSwitch::Value(var)) => {
                    if seen.insert(var.clone()) {
                        used.push(var.clone());
                    }
                }
                Some(ExitSwitch::Fused { args, .. }) => {
                    for a in args {
                        if seen.insert(a.clone()) {
                            used.push(a.clone());
                        }
                    }
                }
                _ => {}
            }
            for link in &b.exits {
                for arg in &link.args {
                    if let LinkArg::Value(var) = arg {
                        if seen.insert(var.clone()) {
                            used.push(var.clone());
                        }
                    }
                }
            }
        }
        for v in used {
            // upstream skips the raw exception metadata variables; in the
            // model graph those travel through `Link.last_exception` /
            // `last_exc_value` (not collected above), so this filter is a
            // defensive parity guard.
            let prefix = v.name_prefix();
            if prefix == "last_exception_" || prefix == "last_exc_value_" {
                continue;
            }
            pending.push((id, v));
        }
    }

    // Already in SSI form (the common case for graphs whose blocks all
    // define what they use): nothing to thread, no snapshot needed.
    if pending.is_empty() {
        return;
    }

    // RPython only runs `SSA_to_SSI` on well-formed graphs, where every
    // used variable is defined on all incoming paths, so threading always
    // terminates at the defining block.  pyre, however, reaches this pass
    // with graphs lowered from Charon MIR that can carry a genuinely
    // undefined operand (the build logs them as "adapter invariant broken
    // … every referenced operand must be defined as a block inputarg or op
    // result"); threading such a variable walks off the top of the graph to
    // the startblock, which has no incoming link to receive it.  Upstream
    // (`rpython/translator/backendopt/ssa.py` `SSA_to_SSI`, which does
    // `del entrymap[graph.startblock]` at :142) would then `KeyError` on
    // `links = entrymap[block]` at :186; pyre instead treats that as the
    // signal that this graph violates the SSI precondition and leaves it
    // exactly as it was — the same already-degenerate jitcode the previous
    // (pre-threading) pipeline produced, with no regression.  This is a
    // migration accommodation, not a permanent structural divergence: pyre
    // must build `FunctionGraph`s from a Rust interpreter (the codewriter
    // never sees interpreter source, unlike upstream — `front/mod.rs`), and
    // is mid-cutover from the transitional legacy rtyper adapters
    // (`translator/rtyper/legacy_*`, slated for retirement — `lib.rs`) to a
    // real-rtyper-typed Charon MIR frontend.  While that cutover is
    // incomplete some graphs are still typed with an undefined operand, so
    // degrading this one graph beats aborting the whole build; the bail tends
    // toward dead code as real-rtyper comes to type every production graph.
    // Well-formed graphs (e.g. `w_list_append`) thread to completion and are
    // kept.
    let snapshot = graph.clone();
    let mut bailed = false;

    while let Some((id, v)) = pending.pop() {
        let v_rep = families.find_rep(v.clone());
        let created = variables_created_in(graph, id);
        if created.contains(&v) {
            continue;
        }
        // Reuse a same-family value the block already defines.  Search a
        // deterministic order (inputargs then op results) rather than the
        // `HashSet` so codegen output is stable.
        let mut matched: Option<Variable> = None;
        {
            let b = graph.block(id);
            'find: for w in b
                .inputargs
                .iter()
                .chain(b.operations.iter().filter_map(|op| op.result.as_ref()))
            {
                if families.find_rep(w.clone()) == v_rep {
                    matched = Some(w.clone());
                    break 'find;
                }
            }
        }
        if let Some(w) = matched {
            renamevariables(graph, id, &v, &w);
        } else {
            let Some(links) = entrymap.get(&id).cloned() else {
                // Undefined operand: threading reached a block with no
                // incoming link.  Abandon the transform for this graph.
                bailed = true;
                break;
            };
            let w = v.copy();
            families.union(v.clone(), w.clone());
            renamevariables(graph, id, &v, &w);
            graph.block_mut(id).inputargs.push(w);
            for (src, idx) in links {
                graph.block_mut(src).exits[idx]
                    .args
                    .push(LinkArg::Value(v.clone()));
                pending.push((src, v.clone()));
            }
        }
    }

    if bailed {
        *graph = snapshot;
    }
}
