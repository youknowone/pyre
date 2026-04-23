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

    let mut vars_by_id: HashMap<u64, (Variable, bool)> = HashMap::new();
    let mut order: Vec<u64> = Vec::new();
    let mut record_var = |var: &Variable, keep: bool| {
        let id = var.id();
        match vars_by_id.get_mut(&id) {
            Some((_existing, existing_keep)) => {
                *existing_keep |= keep;
            }
            None => {
                vars_by_id.insert(id, (var.clone(), keep));
                order.push(id);
            }
        }
    };

    for arg in &link_args {
        if let Some(Hlvalue::Variable(v)) = arg {
            record_var(v, true);
        }
    }
    for op in &newops {
        for arg in &op.args {
            if let Hlvalue::Variable(v) = arg {
                record_var(v, true);
            }
        }
        if let Hlvalue::Variable(v) = &op.result {
            record_var(v, false);
        }
    }

    let vars: Vec<Variable> = order
        .into_iter()
        .filter_map(|id| vars_by_id.remove(&id))
        .filter_map(|(var, keep)| keep.then_some(var))
        .collect();

    let mut mapping: HashMap<Variable, Variable> = HashMap::new();
    for v in &vars {
        mapping.insert(v.clone(), v.copy());
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
