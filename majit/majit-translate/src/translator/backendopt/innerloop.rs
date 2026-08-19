//! Port of `rpython/translator/backendopt/innerloop.py`.
//!
//! Optional support code for backends: it finds which cycles in a graph
//! are likely to correspond to source-level 'inner loops'.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use crate::flowspace::model::{BlockRef, ConcretetypePlaceholder, FunctionGraph, Hlvalue, LinkRef};
use crate::tool::algo::graphlib::{Edge, all_cycles, make_edge_dict};
use crate::tool::algo::unionfind::UnionFind;
use crate::translator::backendopt::ssa::DataFlowFamilyBuilder;

/// Vertex key over a `BlockRef` using Python `is`-identity, mirroring
/// upstream's use of blocks as plain dict keys (`graphlib` keys vertices
/// by object identity). `Rc::ptr_eq` is the analogue of `is`.
#[derive(Clone)]
struct BlockVertex(BlockRef);

impl PartialEq for BlockVertex {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for BlockVertex {}

impl Hash for BlockVertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Rc::as_ptr(&self.0) as *const ()).hash(state);
    }
}

/// `class Loop` at `innerloop.py:10-14`.
pub struct Loop {
    pub headblock: BlockRef,
    /// list of Links making the cycle, starting from one of the exits
    /// of `headblock`.
    pub links: Vec<LinkRef>,
}

/// `find_inner_loops(graph, check_exitswitch_type=None)` at
/// `innerloop.py:17-120`.
///
/// Enumerate what look like the innermost loops of the graph. Returns a
/// list of non-overlapping [`Loop`] instances.
///
/// The heuristic (`innerloop.py:21-39`): of the cycles found, prefer the
/// one with more variables that stay constant across the whole cycle
/// (an inner loop sits inside loops with fewer constants), breaking ties
/// toward fewer blocks; the head is the first `Bool`-switching two-exit
/// block closest to the start block.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn find_inner_loops(
    graph: &FunctionGraph,
    check_exitswitch_type: Option<&ConcretetypePlaceholder>,
) -> Vec<Loop> {
    // `startdistance` = {block: distance-from-startblock}
    // (`innerloop.py:41-56`).
    let mut startdistance: HashMap<BlockVertex, usize> = HashMap::new();
    let mut pending: Vec<BlockRef> = vec![graph.startblock.clone()];
    let mut edge_list: Vec<Rc<Edge<BlockVertex, LinkRef>>> = Vec::new();
    let mut dist = 0usize;
    while !pending.is_empty() {
        let mut newblocks: Vec<BlockRef> = Vec::new();
        for block in &pending {
            if let std::collections::hash_map::Entry::Vacant(e) =
                startdistance.entry(BlockVertex(block.clone()))
            {
                e.insert(dist);
                for link in &block.borrow().exits {
                    let target = link
                        .borrow()
                        .target
                        .clone()
                        .expect("innerloop.py:50 link.target must be set");
                    newblocks.push(target.clone());
                    // Upstream `edge = Edge(block, link.target);
                    // edge.link = link` (`:52-53`).
                    edge_list.push(Rc::new(Edge::with_payload(
                        BlockVertex(block.clone()),
                        BlockVertex(target),
                        link.clone(),
                    )));
                }
            }
        }
        dist += 1;
        pending = newblocks;
    }

    // `vertices = startdistance` (`:58`): the `{block: distance}` dict is
    // handed straight to `all_cycles`, which takes any `VertexSet`.
    let vertices = &startdistance;
    let edges = make_edge_dict(edge_list);
    let cycles = all_cycles(&BlockVertex(graph.startblock.clone()), vertices, &edges);

    let mut loops: Vec<((i64, usize), Loop)> = Vec::new();
    // `variable_families = None` — built lazily on the first cycle that
    // produces a head (`:62,82-84`).
    let mut variable_families: Option<UnionFind<Hlvalue, ()>> = None;

    for cycle in &cycles {
        // find the headblock (`:65-78`).
        let mut candidates: Vec<(usize, usize)> = Vec::new();
        for (i, vertex) in cycle.iter().enumerate() {
            let block = &vertex.source.0;
            let b = block.borrow();
            // `if isinstance(v, Variable) and len(block.exits) == 2:`
            if let Some(Hlvalue::Variable(v)) = &b.exitswitch {
                // `if getattr(v, 'concretetype', None) is
                // check_exitswitch_type:` (`innerloop.py:71`).
                //
                // PRE-EXISTING-ADAPTATION: upstream compares the concrete
                // type by `is` identity; `LowLevelType` (lltype.rs) is a
                // value enum with a structural `PartialEq` (lltype.rs) and
                // carries no identity handle, so this uses `==`. The
                // divergence is currently latent: every caller passes a
                // singleton primitive — the default `None`, and `funcgen.py:99`
                // (`Bool`) which is the sole non-test caller — for which
                // structural `==` and identity `is` coincide. Only a
                // hypothetical non-singleton struct/ptr/forward-ref argument
                // could admit a head that `is` would reject. Convergence path:
                // an lltype identity carrier (same epic as the rtyper
                // lltype-identity model, GH #131).
                if b.exits.len() == 2 && v.concretetype().as_ref() == check_exitswitch_type {
                    candidates.push((startdistance[&BlockVertex(block.clone())], i));
                }
            }
        }
        if candidates.is_empty() {
            continue;
        }
        // `_, i = min(candidates)` (`:76`).
        let (_, i) = *candidates.iter().min().unwrap();
        // `links = [edge.link for edge in cycle[i:] + cycle[:i]]` (`:77`).
        let mut links: Vec<LinkRef> = Vec::new();
        for e in &cycle[i..] {
            links.push(e.payload.clone());
        }
        for e in &cycle[..i] {
            links.push(e.payload.clone());
        }
        let loop_ = Loop {
            headblock: cycle[i].source.0.clone(),
            links,
        };

        // count the variables that remain constant across the cycle,
        // detected as having its SSA family present across all blocks
        // (`:80-99`).
        if variable_families.is_none() {
            variable_families = Some(DataFlowFamilyBuilder::new(graph).into_variable_families());
        }
        let vf = variable_families.as_mut().unwrap();

        let mut num_loop_constants = 0i64;
        // `Block.inputargs` is `Vec<Hlvalue>`, the faithful port of upstream
        // `self.inputargs = list(inputargs)  # mixed list of variable/const`
        // (model.py:176): the stored list is untyped, and `checkgraph`'s
        // `definevar` (model.py:586 `assert isinstance(v, Variable)`, mirrored
        // in `flowspace/model.rs`'s `checkgraph`) enforces the
        // Variable-only shape only on
        // well-formed graphs. Iterating it (`innerloop.py:87`) and handing
        // each element to `find_rep` mirrors upstream's untyped `find_rep`
        // over the same list; a non-Variable — excluded by checkgraph — would
        // become its own singleton family either way.
        let head_inputargs: Vec<Hlvalue> = loop_.headblock.borrow().inputargs.clone();
        for v in &head_inputargs {
            let vrep = vf.find_rep(v.clone());
            // `for link in loop.links: ... else: # found in all blocks`
            let mut found_in_all = true;
            for link in &loop_.links {
                let block1 = link
                    .borrow()
                    .target
                    .clone()
                    .expect("innerloop.py:90 link.target must be set");
                let block1_inputargs: Vec<Hlvalue> = block1.borrow().inputargs.clone();
                let mut found_here = false;
                for v1 in &block1_inputargs {
                    if vf.find_rep(v1.clone()) == vrep {
                        found_here = true;
                        break;
                    }
                }
                if !found_here {
                    found_in_all = false;
                    break;
                }
            }
            if found_in_all {
                num_loop_constants += 1;
            }
        }

        // smaller keys are "better": maximize num_loop_constants, then
        // minimize len(cycle) (`:101-104`).
        let key = (-num_loop_constants, cycle.len());
        loops.push((key, loop_));
    }

    // `loops.sort()` sorts `(key, loop)` tuples (`:106`). On a key tie
    // the tuple compare falls through to the `Loop` objects, which under
    // the Python 2 host order by object id — non-deterministic and a
    // `TypeError` under Python 3. That id-order can't be replicated; this
    // stable sort on the key alone instead keeps the `all_cycles`
    // discovery order on ties, the deterministic part of the pipeline.
    // Distinct (non-overlapping) loops are unaffected; only the kept loop
    // among overlapping equal-key loops can differ from a given host run.
    loops.sort_by_key(|a| a.0);

    // return 'loops' without overlapping blocks (`:108-120`).
    let mut result: Vec<Loop> = Vec::new();
    // `blocks_seen = {}` used as a set of targets (`:110`).
    let mut blocks_seen: HashMap<BlockVertex, bool> = HashMap::new();
    for (_key, loop_) in loops {
        let mut overlapping = false;
        for link in &loop_.links {
            let target = link
                .borrow()
                .target
                .clone()
                .expect("innerloop.py:113 link.target must be set");
            // `if link.target in blocks_seen:` (`:113`).
            if blocks_seen.contains_key(&BlockVertex(target)) {
                overlapping = true;
                break;
            }
        }
        if !overlapping {
            for link in &loop_.links {
                let target = link
                    .borrow()
                    .target
                    .clone()
                    .expect("innerloop.py:118 link.target must be set");
                // `blocks_seen[link.target] = True` (`:119`).
                blocks_seen.insert(BlockVertex(target), true);
            }
            result.push(loop_);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{
        Block, BlockRefExt, ConstValue, Constant, FunctionGraph, Hlvalue, Link, SpaceOperation,
        Variable,
    };

    fn const0() -> Hlvalue {
        Hlvalue::Constant(Constant::new(ConstValue::Int(0)))
    }

    fn hv(v: &Variable) -> Hlvalue {
        Hlvalue::Variable(v.clone())
    }

    fn op(name: &str, args: Vec<Hlvalue>, result: &Variable) -> SpaceOperation {
        SpaceOperation::new(name, args, Hlvalue::Variable(result.clone()))
    }

    /// `link in block.exits` — identity membership (`test_innerloop.py:19`).
    fn link_in_exits(link: &LinkRef, block: &BlockRef) -> bool {
        block.borrow().exits.iter().any(|l| Rc::ptr_eq(l, link))
    }

    #[test]
    fn straight_line_graph_has_no_inner_loops() {
        // start -> done, no back-edge.
        let start = Block::shared(vec![]);
        let done = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());
        start.closeblock(vec![Link::new(vec![], Some(done.clone()), None).into_ref()]);

        assert!(find_inner_loops(&graph, None).is_empty());
    }

    #[test]
    fn single_loop_is_found_with_head_at_the_switch_block() {
        // start -> head(i); head switches (2 exits): -> body, -> done;
        // body -> head (back-edge). The inner loop is head -> body.
        let i_head = Variable::new();
        let head = Block::shared(vec![Hlvalue::Variable(i_head.clone())]);
        let i_body = Variable::new();
        let body = Block::shared(vec![Hlvalue::Variable(i_body.clone())]);
        let done = Block::shared(vec![]);
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());

        // start -> head, seeding i_head with a constant.
        start.closeblock(vec![
            Link::new(vec![const0()], Some(head.clone()), None).into_ref(),
        ]);

        // head switches on a Variable with two exits.
        let cond = Variable::new();
        head.borrow_mut().exitswitch = Some(Hlvalue::Variable(cond));
        let head_to_body = Link::new(
            vec![Hlvalue::Variable(i_head.clone())],
            Some(body.clone()),
            None,
        )
        .into_ref();
        let head_to_done = Link::new(vec![], Some(done.clone()), None).into_ref();
        head.closeblock(vec![head_to_body, head_to_done]);

        // body -> head (the back-edge), threading i_body back into i_head.
        body.closeblock(vec![
            Link::new(vec![Hlvalue::Variable(i_body)], Some(head.clone()), None).into_ref(),
        ]);

        let loops = find_inner_loops(&graph, None);
        assert_eq!(loops.len(), 1);
        // head is the Bool-switching two-exit block, so it heads the loop.
        assert!(Rc::ptr_eq(&loops[0].headblock, &head));
        // cycle is head -> body -> head: two links.
        assert_eq!(loops[0].links.len(), 2);
    }

    /// Parity with `test_innerloop.py:23 test_two_loops`: two sequential
    /// `while` loops are returned as two non-overlapping `Loop`s.
    #[test]
    fn two_sequential_loops_are_both_found() {
        let (x1, y1) = (Variable::new(), Variable::new());
        let head1 = Block::shared(vec![hv(&x1), hv(&y1)]);
        let (xb1, yb1) = (Variable::new(), Variable::new());
        let body1 = Block::shared(vec![hv(&xb1), hv(&yb1)]);
        let (x2, y2) = (Variable::new(), Variable::new());
        let head2 = Block::shared(vec![hv(&x2), hv(&y2)]);
        let (xb2, yb2) = (Variable::new(), Variable::new());
        let body2 = Block::shared(vec![hv(&xb2), hv(&yb2)]);
        let yd = Variable::new();
        let done = Block::shared(vec![hv(&yd)]);
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());

        // head1: while y1 > 0 (op "gt"), switching.
        let cond1 = Variable::new();
        {
            let mut b = head1.borrow_mut();
            b.operations.push(op("gt", vec![hv(&y1), const0()], &cond1));
            b.exitswitch = Some(hv(&cond1));
        }
        // body1: y -= x (op "sub" producing a new y).
        let ynew1 = Variable::new();
        body1
            .borrow_mut()
            .operations
            .push(op("sub", vec![hv(&yb1), hv(&xb1)], &ynew1));
        // head2: while y2 < 0 (op "lt"), switching.
        let cond2 = Variable::new();
        {
            let mut b = head2.borrow_mut();
            b.operations.push(op("lt", vec![hv(&y2), const0()], &cond2));
            b.exitswitch = Some(hv(&cond2));
        }
        // body2: y += x (op "add" producing a new y).
        let ynew2 = Variable::new();
        body2
            .borrow_mut()
            .operations
            .push(op("add", vec![hv(&yb2), hv(&xb2)], &ynew2));

        start.closeblock(vec![
            Link::new(vec![const0(), const0()], Some(head1.clone()), None).into_ref(),
        ]);
        // exits[0] is the loop body, exits[1] leaves to the next loop.
        head1.closeblock(vec![
            Link::new(vec![hv(&x1), hv(&y1)], Some(body1.clone()), None).into_ref(),
            Link::new(vec![hv(&x1), hv(&y1)], Some(head2.clone()), None).into_ref(),
        ]);
        body1.closeblock(vec![
            Link::new(vec![hv(&xb1), hv(&ynew1)], Some(head1.clone()), None).into_ref(),
        ]);
        head2.closeblock(vec![
            Link::new(vec![hv(&x2), hv(&y2)], Some(body2.clone()), None).into_ref(),
            Link::new(vec![hv(&y2)], Some(done.clone()), None).into_ref(),
        ]);
        body2.closeblock(vec![
            Link::new(vec![hv(&xb2), hv(&ynew2)], Some(head2.clone()), None).into_ref(),
        ]);

        let loops = find_inner_loops(&graph, None);
        assert_eq!(loops.len(), 2);
        assert!(!Rc::ptr_eq(&loops[0].headblock, &loops[1].headblock));
        for loop_ in &loops {
            let head = &loop_.headblock;
            let opname = head.borrow().operations[0].opname.clone();
            assert!(opname == "gt" || opname == "lt");
            assert_eq!(loop_.links.len(), 2);
            // links[0] is one of the head's exits; links[1] returns to it.
            assert!(link_in_exits(&loop_.links[0], head));
            assert!(Rc::ptr_eq(
                loop_.links[1].borrow().target.as_ref().unwrap(),
                head
            ));
        }
    }

    /// Parity with `test_innerloop.py:44 test_nested_loops`: the inner
    /// loop carries more loop-constant families, so it sorts first and the
    /// overlapping outer loop is dropped — only the inner loop is returned.
    #[test]
    fn nested_loops_return_only_the_inner_loop() {
        let (yo, zo) = (Variable::new(), Variable::new());
        let head_outer = Block::shared(vec![hv(&yo), hv(&zo)]);
        let (yi, zi) = (Variable::new(), Variable::new());
        let head_inner = Block::shared(vec![hv(&yi), hv(&zi)]);
        let (yb, zb) = (Variable::new(), Variable::new());
        let body_inner = Block::shared(vec![hv(&yb), hv(&zb)]);
        let (ya, za) = (Variable::new(), Variable::new());
        let after_inner = Block::shared(vec![hv(&ya), hv(&za)]);
        let zd = Variable::new();
        let done = Block::shared(vec![hv(&zd)]);
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());

        // head_outer: while y <= 10 (op "le"), switching.
        let cond_o = Variable::new();
        {
            let mut b = head_outer.borrow_mut();
            b.operations
                .push(op("le", vec![hv(&yo), const0()], &cond_o));
            b.exitswitch = Some(hv(&cond_o));
        }
        // head_inner: while z < y (op "lt"), switching.
        let cond_i = Variable::new();
        {
            let mut b = head_inner.borrow_mut();
            b.operations.push(op("lt", vec![hv(&zi), hv(&yi)], &cond_i));
            b.exitswitch = Some(hv(&cond_i));
        }
        // body_inner: z = z + y — a fresh z breaks head_inner.z into a
        // true phi, so z is NOT a loop constant of the outer loop.
        let z_new = Variable::new();
        body_inner
            .borrow_mut()
            .operations
            .push(op("add", vec![hv(&zb), hv(&yb)], &z_new));
        // after_inner: y = y + 1 — a fresh y on the outer back-edge.
        let y_new = Variable::new();
        after_inner
            .borrow_mut()
            .operations
            .push(op("add", vec![hv(&ya), const0()], &y_new));

        start.closeblock(vec![
            Link::new(vec![const0(), const0()], Some(head_outer.clone()), None).into_ref(),
        ]);
        head_outer.closeblock(vec![
            Link::new(vec![hv(&yo), hv(&zo)], Some(head_inner.clone()), None).into_ref(),
            Link::new(vec![hv(&zo)], Some(done.clone()), None).into_ref(),
        ]);
        head_inner.closeblock(vec![
            Link::new(vec![hv(&yi), hv(&zi)], Some(body_inner.clone()), None).into_ref(),
            Link::new(vec![hv(&yi), hv(&zi)], Some(after_inner.clone()), None).into_ref(),
        ]);
        body_inner.closeblock(vec![
            Link::new(vec![hv(&yb), hv(&z_new)], Some(head_inner.clone()), None).into_ref(),
        ]);
        after_inner.closeblock(vec![
            Link::new(vec![hv(&y_new), hv(&za)], Some(head_outer.clone()), None).into_ref(),
        ]);

        let loops = find_inner_loops(&graph, None);
        assert_eq!(loops.len(), 1);
        assert!(Rc::ptr_eq(&loops[0].headblock, &head_inner));
        assert_eq!(loops[0].headblock.borrow().operations[0].opname, "lt");
        assert_eq!(loops[0].links.len(), 2);
    }
}
