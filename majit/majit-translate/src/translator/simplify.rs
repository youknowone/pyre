//! RPython `rpython/translator/simplify.py` — graph-level transformations
//! invoked from `RPythonAnnotator.simplify()` (annrpython.py:336-371)
//! and `simplify_graph` (simplify.py:1075+).
//!
//! Only the subset reachable from the annotator port lands here
//! initially; downstream simplification passes arrive with the
//! rtyper port.

use std::collections::{HashMap, HashSet};

use crate::flowspace::model::{
    BlockKey, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, Hlvalue, LinkRef,
    SpaceOperation, Variable, mkentrymap,
};

/// RPython `eliminate_empty_blocks(graph)` (simplify.py:52-69).
///
/// ```python
/// def eliminate_empty_blocks(graph):
///     """Eliminate basic blocks that do not contain any operations.
///     When this happens, we need to replace the preceeding link with
///     the following link.  Arguments of the links should be updated."""
///     for link in list(graph.iterlinks()):
///         while not link.target.operations:
///             block1 = link.target
///             if block1.exitswitch is not None:
///                 break
///             if not block1.exits:
///                 break
///             exit = block1.exits[0]
///             assert block1 is not exit.target, (
///                 "the graph contains an empty infinite loop")
///             subst = dict(zip(block1.inputargs, link.args))
///             link.args = [v.replace(subst) for v in exit.args]
///             link.target = exit.target
/// ```
///
/// The Rust port mirrors upstream's single-threaded mutation pattern:
/// walk each snapshot link, collapse chains of empty successor
/// blocks, and rewrite the link's `args` / `target` in place.
pub fn eliminate_empty_blocks(graph: &FunctionGraph) {
    // upstream: `for link in list(graph.iterlinks()):`
    for link_ref in graph.iterlinks() {
        loop {
            // upstream: `while not link.target.operations:`
            let (is_empty, has_switch, has_exits, next_exit, target_rc) = {
                let link = link_ref.borrow();
                let Some(target) = link.target.as_ref() else {
                    break;
                };
                let target_b = target.borrow();
                let is_empty = target_b.operations.is_empty();
                let has_switch = target_b.exitswitch.is_some();
                let has_exits = !target_b.exits.is_empty();
                let next_exit = target_b.exits.first().cloned();
                (is_empty, has_switch, has_exits, next_exit, target.clone())
            };
            if !is_empty {
                break;
            }
            if has_switch {
                // upstream: `if block1.exitswitch is not None: break`.
                break;
            }
            if !has_exits {
                // upstream: `if not block1.exits: break`.
                break;
            }
            // upstream: `exit = block1.exits[0]`.
            let Some(exit_ref) = next_exit else {
                break;
            };

            // upstream: `assert block1 is not exit.target, ...`.
            {
                let exit = exit_ref.borrow();
                if let Some(exit_target) = &exit.target {
                    assert!(
                        !std::rc::Rc::ptr_eq(exit_target, &target_rc),
                        "the graph contains an empty infinite loop"
                    );
                }
            }

            // Build `subst = dict(zip(block1.inputargs, link.args))` —
            // the Link carries `Vec<LinkArg> = Vec<Option<Hlvalue>>`,
            // so we keep entries with concrete values only and skip
            // the transient-merge `None` slots.
            let subst: HashMap<Variable, Hlvalue> = {
                let target_b = target_rc.borrow();
                let link = link_ref.borrow();
                target_b
                    .inputargs
                    .iter()
                    .zip(link.args.iter())
                    .filter_map(|(formal, actual)| {
                        let Hlvalue::Variable(v) = formal else {
                            return None;
                        };
                        actual.as_ref().map(|a| (v.clone(), a.clone()))
                    })
                    .collect()
            };

            // upstream: `link.args = [v.replace(subst) for v in exit.args]`.
            let new_args: Vec<Option<Hlvalue>> = {
                let exit = exit_ref.borrow();
                exit.args
                    .iter()
                    .map(|arg| {
                        arg.as_ref().map(|v| match v {
                            Hlvalue::Variable(var) => subst
                                .get(var)
                                .cloned()
                                .unwrap_or_else(|| Hlvalue::Variable(var.clone())),
                            other => other.clone(),
                        })
                    })
                    .collect()
            };

            // upstream: `link.target = exit.target`.
            let new_target = exit_ref.borrow().target.clone();

            {
                let mut link = link_ref.borrow_mut();
                link.args = new_args;
                link.target = new_target;
            }
            // upstream: the while-loop re-evaluates on the rewritten
            // link so chains of empty blocks collapse in one pass.
        }
    }
}

/// Inline helper — upstream's `.replace(mapping)` when `mapping` carries
/// `dict[Variable, Hlvalue]`. The Rust port's `Hlvalue.replace` method
/// narrows its signature to `HashMap<Variable, Variable>` because that
/// is all its existing callers need (`copygraph`, `renamevariables`);
/// `join_blocks` needs the upstream-wide semantics (a Variable can map
/// to a Constant when the caller threads through `link.args`). This
/// helper keeps the wide semantics local to `simplify.rs` instead of
/// widening the public API surface.
fn rename_hl(v: &Hlvalue, mapping: &HashMap<Variable, Hlvalue>) -> Hlvalue {
    match v {
        Hlvalue::Variable(var) => mapping
            .get(var)
            .cloned()
            .unwrap_or_else(|| Hlvalue::Variable(var.clone())),
        Hlvalue::Constant(_) => v.clone(),
    }
}

fn rename_op(op: &SpaceOperation, mapping: &HashMap<Variable, Hlvalue>) -> SpaceOperation {
    // upstream: `op = op.replace(renaming)`.
    let mut new_op = SpaceOperation {
        opname: op.opname.clone(),
        args: op.args.iter().map(|a| rename_hl(a, mapping)).collect(),
        result: rename_hl(&op.result, mapping),
        offset: op.offset,
    };
    // upstream: indirect_call with a Constant callee is rewritten back
    // to direct_call (simplify.py:293-297).
    if new_op.opname == "indirect_call" {
        if matches!(new_op.args.first(), Some(Hlvalue::Constant(_))) {
            assert!(
                matches!(new_op.args.last(), Some(Hlvalue::Constant(_))),
                "indirect_call's trailing argument must be a Constant"
            );
            new_op.args.pop();
            new_op.opname = "direct_call".to_string();
        }
    }
    new_op
}

fn rename_link_args(link: &LinkRef, mapping: &HashMap<Variable, Hlvalue>) -> LinkRef {
    // upstream's `link.replace(mapping)` path — rebuild a fresh Link
    // with renamed args / last_exception / last_exc_value.
    let src = link.borrow();
    let mut newlink = crate::flowspace::model::Link::new_mergeable(
        src.args
            .iter()
            .map(|a| a.as_ref().map(|v| rename_hl(v, mapping)))
            .collect(),
        src.target.clone(),
        src.exitcase.clone(),
    );
    newlink.prevblock = src.prevblock.clone();
    newlink.llexitcase = src.llexitcase.clone();
    newlink.last_exception = src.last_exception.as_ref().map(|v| rename_hl(v, mapping));
    newlink.last_exc_value = src.last_exc_value.as_ref().map(|v| rename_hl(v, mapping));
    newlink.into_ref()
}

/// RPython `replace_exitswitch_by_constant(block, const)`
/// (simplify.py:36-48).
///
/// ```python
/// def replace_exitswitch_by_constant(block, const):
///     newexits = [link for link in block.exits
///                      if link.exitcase == const.value]
///     if len(newexits) == 0:
///         newexits = [link for link in block.exits
///                      if link.exitcase == 'default']
///     assert len(newexits) == 1
///     newexits[0].exitcase = None
///     if hasattr(newexits[0], 'llexitcase'):
///         newexits[0].llexitcase = None
///     block.exitswitch = None
///     block.recloseblock(*newexits)
///     return newexits
/// ```
pub fn replace_exitswitch_by_constant(block: &BlockRef, const_: &Constant) -> Vec<LinkRef> {
    // upstream: `if link.exitcase == const.value` — our `Link.exitcase`
    // is `Option<Hlvalue>` wrapping a Constant carrying `ConstValue`;
    // compare on the inner `ConstValue`.
    let cases_eq = |link: &LinkRef| match &link.borrow().exitcase {
        Some(Hlvalue::Constant(c)) => c.value == const_.value,
        _ => false,
    };
    let default_case = |link: &LinkRef| {
        matches!(
            &link.borrow().exitcase,
            Some(Hlvalue::Constant(c)) if matches!(&c.value, ConstValue::Str(s) if s == "default")
        )
    };

    let exits_snapshot: Vec<LinkRef> = block.borrow().exits.iter().cloned().collect();
    let mut newexits: Vec<LinkRef> = exits_snapshot
        .iter()
        .filter(|l| cases_eq(l))
        .cloned()
        .collect();
    if newexits.is_empty() {
        newexits = exits_snapshot
            .iter()
            .filter(|l| default_case(l))
            .cloned()
            .collect();
    }
    assert_eq!(
        newexits.len(),
        1,
        "replace_exitswitch_by_constant: no unique surviving exit"
    );
    {
        let mut l = newexits[0].borrow_mut();
        l.exitcase = None;
        l.llexitcase = None;
    }
    block.borrow_mut().exitswitch = None;
    block.recloseblock(newexits.clone());
    newexits
}

/// RPython `join_blocks(graph)` (simplify.py:271-319).
///
/// ```python
/// def join_blocks(graph):
///     """Links can be deleted if they are the single exit of a block and
///     the single entry point of the next block.  When this happens, we can
///     append all the operations of the following block to the preceeding
///     block (but renaming variables with the appropriate arguments.)
///     """
///     entrymap = mkentrymap(graph)
///     block = graph.startblock
///     seen = {block: True}
///     stack = list(block.exits)
///     while stack:
///         link = stack.pop()
///         if (link.prevblock.exitswitch is None and
///             len(entrymap[link.target]) == 1 and
///             link.target.exits):  # stop at the returnblock
///             assert len(link.prevblock.exits) == 1
///             renaming = {}
///             for vprev, vtarg in zip(link.args, link.target.inputargs):
///                 renaming[vtarg] = vprev
///             def rename_op(op):
///                 op = op.replace(renaming)
///                 # special case...
///                 if op.opname == 'indirect_call':
///                     if isinstance(op.args[0], Constant):
///                         assert isinstance(op.args[-1], Constant)
///                         del op.args[-1]
///                         op.opname = 'direct_call'
///                 return op
///             for op in link.target.operations:
///                 link.prevblock.operations.append(rename_op(op))
///             exits = []
///             for exit in link.target.exits:
///                 newexit = exit.replace(renaming)
///                 exits.append(newexit)
///             if link.target.exitswitch:
///                 newexitswitch = link.target.exitswitch.replace(renaming)
///             else:
///                 newexitswitch = None
///             link.prevblock.exitswitch = newexitswitch
///             link.prevblock.recloseblock(*exits)
///             if (isinstance(newexitswitch, Constant) and
///                     not link.prevblock.canraise):
///                 exits = replace_exitswitch_by_constant(link.prevblock,
///                                                        newexitswitch)
///             stack.extend(exits)
///         else:
///             if link.target not in seen:
///                 stack.extend(link.target.exits)
///                 seen[link.target] = True
/// ```
pub fn join_blocks(graph: &FunctionGraph) {
    let entrymap = mkentrymap(graph);
    let mut seen: HashSet<BlockKey> = HashSet::new();
    seen.insert(BlockKey::of(&graph.startblock));

    // upstream: `stack = list(block.exits)`.
    let mut stack: Vec<LinkRef> = graph.startblock.borrow().exits.iter().cloned().collect();

    while let Some(link) = stack.pop() {
        // Snapshot the fields we need before re-borrowing the Link
        // itself (Rust borrow-checker keeps mutation and reads split).
        let (prev_rc, target_rc, target_inputargs, link_args_all_some, link_args) = {
            let l = link.borrow();
            let prev = l.prevblock.as_ref().and_then(|w| w.upgrade());
            let target = l.target.clone();
            let target_inputargs = target
                .as_ref()
                .map(|t| t.borrow().inputargs.clone())
                .unwrap_or_default();
            let args_all_some = l.args.iter().all(|a| a.is_some());
            let args: Vec<Hlvalue> = l.args.iter().filter_map(|a| a.clone()).collect();
            (prev, target, target_inputargs, args_all_some, args)
        };
        let Some(prev_rc) = prev_rc else {
            continue;
        };
        let Some(target_rc) = target_rc else {
            continue;
        };

        let prev_switch_is_none = prev_rc.borrow().exitswitch.is_none();
        let target_entry_count = entrymap
            .get(&BlockKey::of(&target_rc))
            .map(Vec::len)
            .unwrap_or(0);
        let target_has_exits = !target_rc.borrow().exits.is_empty();

        // upstream: `link.prevblock.exitswitch is None and
        // len(entrymap[link.target]) == 1 and link.target.exits`.
        let can_join = prev_switch_is_none
            && target_entry_count == 1
            && target_has_exits
            && link_args_all_some;

        if can_join {
            assert_eq!(
                prev_rc.borrow().exits.len(),
                1,
                "join_blocks: prevblock has exitswitch=None but != 1 exit"
            );

            // upstream: `renaming[vtarg] = vprev` for each pair.
            let mut renaming: HashMap<Variable, Hlvalue> = HashMap::new();
            for (vprev, vtarg) in link_args.iter().zip(target_inputargs.iter()) {
                if let Hlvalue::Variable(tgt) = vtarg {
                    renaming.insert(tgt.clone(), vprev.clone());
                }
            }

            // upstream: append renamed target.operations to prevblock.
            let target_ops: Vec<SpaceOperation> = target_rc.borrow().operations.clone();
            for op in &target_ops {
                prev_rc
                    .borrow_mut()
                    .operations
                    .push(rename_op(op, &renaming));
            }

            // upstream: `exits = [exit.replace(renaming) for exit in target.exits]`.
            let target_exits: Vec<LinkRef> = target_rc.borrow().exits.iter().cloned().collect();
            let mut new_exits: Vec<LinkRef> = target_exits
                .iter()
                .map(|e| rename_link_args(e, &renaming))
                .collect();

            // upstream: `newexitswitch = target.exitswitch.replace(renaming) or None`.
            let new_exitswitch = target_rc
                .borrow()
                .exitswitch
                .as_ref()
                .map(|sw| rename_hl(sw, &renaming));
            prev_rc.borrow_mut().exitswitch = new_exitswitch.clone();
            prev_rc.recloseblock(new_exits.clone());

            // upstream: constant-fold the new switch when prevblock is
            // not a can-raise block (simplify.py:311-314).
            if let Some(Hlvalue::Constant(const_)) = &new_exitswitch {
                if !prev_rc.borrow().canraise() {
                    new_exits = replace_exitswitch_by_constant(&prev_rc, const_);
                }
            }

            stack.extend(new_exits);
        } else {
            // upstream: `if link.target not in seen: stack.extend(link.target.exits)`.
            let target_key = BlockKey::of(&target_rc);
            if !seen.contains(&target_key) {
                seen.insert(target_key);
                let more: Vec<LinkRef> = target_rc.borrow().exits.iter().cloned().collect();
                stack.extend(more);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{Block, FunctionGraph, Hlvalue, Link, Variable};
    use std::rc::Rc;

    #[test]
    fn elimination_collapses_single_empty_block() {
        // start -> empty (no ops, 1 formal arg) -> return
        // empty.exits[0].args = [empty.inputargs[0]]
        // After: start.exits[0] rewrites to point at return with the
        // substituted arg from start's link.args.
        let formal = Variable::new();
        let empty = Block::shared(vec![Hlvalue::Variable(formal.clone())]);
        let start = Block::shared(vec![]);

        let graph = FunctionGraph::new("f", start.clone());
        let returnblock = graph.returnblock.clone();

        // start -> empty with args [int(42)]
        let actual = Hlvalue::Constant(crate::flowspace::model::Constant::new(
            crate::flowspace::model::ConstValue::Int(42),
        ));
        let start_link = Link::new(vec![actual], Some(empty.clone()), None).into_ref();
        start.borrow_mut().closeblock(vec![start_link]);

        // empty -> return with args [formal]
        let empty_link = Link::new(
            vec![Hlvalue::Variable(formal.clone())],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        empty.borrow_mut().closeblock(vec![empty_link]);

        eliminate_empty_blocks(&graph);

        // start -> returnblock now (empty collapsed).
        let s = start.borrow();
        let exit = s.exits[0].borrow();
        assert!(Rc::ptr_eq(exit.target.as_ref().unwrap(), &returnblock));
        // And the link args carry the substituted constant.
        assert!(matches!(
            exit.args[0].as_ref(),
            Some(Hlvalue::Constant(c)) if matches!(c.value, crate::flowspace::model::ConstValue::Int(42))
        ));
    }

    #[test]
    fn elimination_keeps_blocks_with_operations() {
        // start -> body (has an op) -> return — body is NOT empty.
        let v_in = Variable::new();
        let v_out = Variable::new();
        let body = Block::shared(vec![Hlvalue::Variable(v_in.clone())]);
        body.borrow_mut()
            .operations
            .push(crate::flowspace::model::SpaceOperation::new(
                "op".to_string(),
                vec![Hlvalue::Variable(v_in.clone())],
                Hlvalue::Variable(v_out.clone()),
            ));
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());
        let returnblock = graph.returnblock.clone();

        let start_link = Link::new(
            vec![Hlvalue::Variable(Variable::new())],
            Some(body.clone()),
            None,
        )
        .into_ref();
        start.borrow_mut().closeblock(vec![start_link]);

        let body_link = Link::new(
            vec![Hlvalue::Variable(v_out)],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        body.borrow_mut().closeblock(vec![body_link]);

        eliminate_empty_blocks(&graph);

        // start -> body unchanged.
        let s = start.borrow();
        let exit = s.exits[0].borrow();
        assert!(Rc::ptr_eq(exit.target.as_ref().unwrap(), &body));
    }

    #[test]
    fn join_blocks_merges_linear_chain() {
        // start -> mid -> return.
        // `mid` has one entry (from start) and non-empty exits ⇒ its
        // operations get folded into start, and start takes over mid's
        // exit to returnblock.
        let v_in = Variable::new();
        let v_out = Variable::new();
        let mid = Block::shared(vec![Hlvalue::Variable(v_in.clone())]);
        mid.borrow_mut()
            .operations
            .push(crate::flowspace::model::SpaceOperation::new(
                "int_add",
                vec![
                    Hlvalue::Variable(v_in.clone()),
                    Hlvalue::Constant(crate::flowspace::model::Constant::new(
                        crate::flowspace::model::ConstValue::Int(1),
                    )),
                ],
                Hlvalue::Variable(v_out.clone()),
            ));

        let start_in = Variable::new();
        let start = Block::shared(vec![Hlvalue::Variable(start_in.clone())]);
        let graph = FunctionGraph::new("f", start.clone());
        let returnblock = graph.returnblock.clone();

        let start_link = Link::new(
            vec![Hlvalue::Variable(start_in.clone())],
            Some(mid.clone()),
            None,
        )
        .into_ref();
        start.borrow_mut().closeblock(vec![start_link.clone()]);
        start_link.borrow_mut().prevblock = Some(Rc::downgrade(&start));

        let mid_link = Link::new(
            vec![Hlvalue::Variable(v_out.clone())],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        mid.borrow_mut().closeblock(vec![mid_link.clone()]);
        mid_link.borrow_mut().prevblock = Some(Rc::downgrade(&mid));

        join_blocks(&graph);

        // `mid`'s op should now be on `start`, with v_in renamed to
        // start_in in the folded op's args.
        let s = start.borrow();
        assert_eq!(s.operations.len(), 1);
        assert_eq!(s.operations[0].opname, "int_add");
        assert!(matches!(
            &s.operations[0].args[0],
            Hlvalue::Variable(v) if *v == start_in
        ));
        // start's exit should now target returnblock (mid got folded).
        assert_eq!(s.exits.len(), 1);
        assert!(Rc::ptr_eq(
            s.exits[0].borrow().target.as_ref().unwrap(),
            &returnblock
        ));
    }

    #[test]
    fn join_blocks_keeps_diamond() {
        // start -> {a, b} — both target the same block `merge`. `merge`
        // has 2 entries ⇒ join_blocks must NOT fold either into start
        // (entrymap[merge] != 1).
        let v = Variable::new();
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let merge = Block::shared(vec![Hlvalue::Variable(Variable::new())]);
        let graph = FunctionGraph::new("f", start.clone());
        let returnblock = graph.returnblock.clone();

        // Boolean switch on `v`: False -> merge, True -> merge.
        start.borrow_mut().exitswitch = Some(Hlvalue::Variable(v.clone()));
        let left = Link::new(
            vec![Hlvalue::Variable(v.clone())],
            Some(merge.clone()),
            Some(Hlvalue::Constant(crate::flowspace::model::Constant::new(
                crate::flowspace::model::ConstValue::Bool(false),
            ))),
        )
        .into_ref();
        let right = Link::new(
            vec![Hlvalue::Variable(v.clone())],
            Some(merge.clone()),
            Some(Hlvalue::Constant(crate::flowspace::model::Constant::new(
                crate::flowspace::model::ConstValue::Bool(true),
            ))),
        )
        .into_ref();
        start
            .borrow_mut()
            .closeblock(vec![left.clone(), right.clone()]);
        left.borrow_mut().prevblock = Some(Rc::downgrade(&start));
        right.borrow_mut().prevblock = Some(Rc::downgrade(&start));

        let merge_link = Link::new(
            vec![merge.borrow().inputargs[0].clone()],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        merge.borrow_mut().closeblock(vec![merge_link.clone()]);
        merge_link.borrow_mut().prevblock = Some(Rc::downgrade(&merge));

        let ops_before = merge.borrow().operations.len();
        join_blocks(&graph);
        let ops_after = merge.borrow().operations.len();

        // Start still has 2 exits, merge still has 1 op-list, returnblock unchanged.
        assert_eq!(start.borrow().exits.len(), 2);
        assert_eq!(ops_before, ops_after);
    }
}
