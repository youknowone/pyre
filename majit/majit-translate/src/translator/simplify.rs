//! RPython `rpython/translator/simplify.py` — graph-level transformations
//! invoked from `RPythonAnnotator.simplify()` (annrpython.py:336-371)
//! and `simplify_graph` (simplify.py:1075+).
//!
//! Only the subset reachable from the annotator port lands here
//! initially; downstream simplification passes arrive with the
//! rtyper port.

use std::collections::HashMap;

use crate::flowspace::model::{FunctionGraph, Hlvalue, Variable};

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
}
