//! RPython `rpython/translator/unsimplify.py`.

use std::collections::HashMap;

use crate::flowspace::model::{
    Block, BlockRef, BlockRefExt, Hlvalue, Link, LinkRef, SpaceOperation, Variable,
};

/// RPython `unsimplify.insert_empty_block(link, newops=[])`
/// (unsimplify.py:10-31).
pub fn insert_empty_block(link: &LinkRef, newops: Vec<SpaceOperation>) -> BlockRef {
    let link_args = link.borrow().args.clone();
    let target = link
        .borrow()
        .target
        .clone()
        .expect("insert_empty_block requires link.target");

    // `vars = {}` with insertion-ordered iteration to mirror CPython
    // dict semantics. `order` preserves first-insertion order; the
    // HashMap holds the (Variable, keep) pair keyed by Variable id.
    let mut vars_by_id: HashMap<u64, (Variable, bool)> = HashMap::new();
    let mut order: Vec<u64> = Vec::new();

    // `for v in link.args: vars[v] = True` — unconditional assign.
    for arg in &link_args {
        if let Some(Hlvalue::Variable(v)) = arg {
            let id = v.id();
            match vars_by_id.get_mut(&id) {
                Some((_existing, existing_keep)) => {
                    *existing_keep = true;
                }
                None => {
                    vars_by_id.insert(id, (v.clone(), true));
                    order.push(id);
                }
            }
        }
    }
    for op in &newops {
        // `for v in op.args: vars.setdefault(v, True)` — insert True
        // only when absent; never promote an existing False (e.g. a
        // prior op's result) back to True.
        for arg in &op.args {
            if let Hlvalue::Variable(v) = arg {
                let id = v.id();
                if !vars_by_id.contains_key(&id) {
                    vars_by_id.insert(id, (v.clone(), true));
                    order.push(id);
                }
            }
        }
        // `vars[op.result] = False` — hard overwrite, demoting prior
        // True (e.g. from a link.arg or earlier op.arg) to False.
        if let Hlvalue::Variable(v) = &op.result {
            let id = v.id();
            match vars_by_id.get_mut(&id) {
                Some((_existing, existing_keep)) => {
                    *existing_keep = false;
                }
                None => {
                    vars_by_id.insert(id, (v.clone(), false));
                    order.push(id);
                }
            }
        }
    }

    let vars: Vec<Variable> = order
        .into_iter()
        .filter_map(|id| vars_by_id.remove(&id))
        .filter_map(|(var, keep)| keep.then_some(var))
        .collect();

    let mut mapping: HashMap<Variable, Hlvalue> = HashMap::new();
    for v in &vars {
        mapping.insert(v.clone(), Hlvalue::Variable(v.copy()));
    }

    let newblock = Block::shared(vars.iter().cloned().map(Hlvalue::Variable).collect());
    {
        let mut nb = newblock.borrow_mut();
        nb.operations.extend(newops);
    }
    let exit = Link::new_mergeable(link_args.clone(), Some(target), None).into_ref();
    newblock.closeblock(vec![exit]);
    newblock.borrow_mut().renamevariables(&mapping);

    let mut link_mut = link.borrow_mut();
    link_mut.args = vars.into_iter().map(Hlvalue::Variable).map(Some).collect();
    link_mut.target = Some(newblock.clone());
    drop(link_mut);

    newblock
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(name: &str) -> Variable {
        Variable::named(name)
    }

    fn hv(v: &Variable) -> Hlvalue {
        Hlvalue::Variable(v.clone())
    }

    fn make_link(args: Vec<Variable>, target_inputargs: usize) -> LinkRef {
        let target = Block::shared(
            (0..target_inputargs)
                .map(|i| hv(&var(&format!("t{i}"))))
                .collect(),
        );
        Link::new(args.iter().map(hv).collect(), Some(target), None).into_ref()
    }

    /// `vars.setdefault(v, True)` on a variable whose prior `False`
    /// came from an earlier op.result must NOT promote it back to
    /// True. Covers a two-step conversion chain `a → c → d`, where
    /// `c` is an intermediate produced by the first op and consumed
    /// by the second: upstream keeps `c` out of the kept-vars set.
    #[test]
    fn op_arg_does_not_override_earlier_op_result_false_keep_flag() {
        let a = var("a");
        let b = var("b");
        let c = var("c");
        let d = var("d");
        let link = make_link(vec![a.clone(), b.clone()], 2);
        let newops = vec![
            SpaceOperation::new("convert1", vec![hv(&a)], hv(&c)),
            SpaceOperation::new("convert2", vec![hv(&c)], hv(&d)),
        ];

        let newblock = insert_empty_block(&link, newops);

        // Kept vars = [a, b]; `c` (earlier op.result) stays False even
        // after appearing as a later op.arg.
        assert_eq!(newblock.borrow().inputargs.len(), 2);
        assert_eq!(link.borrow().args.len(), 2);
    }

    /// `vars[op.result] = False` must hard-overwrite a prior True
    /// placed by `link.args`. Covers the case where an outgoing-link
    /// variable is reused as the result of an injected op.
    #[test]
    fn op_result_overwrites_link_arg_keep_flag_to_false() {
        let v = var("v");
        let w = var("w");
        let link = make_link(vec![v.clone()], 1);
        let newops = vec![SpaceOperation::new("produce_v", vec![hv(&w)], hv(&v))];

        let newblock = insert_empty_block(&link, newops);

        // `v` was True from link.args then False from op.result; drop
        // it. Only `w` (new from op.args) remains kept.
        assert_eq!(newblock.borrow().inputargs.len(), 1);
        assert_eq!(link.borrow().args.len(), 1);
    }
}
