//! Strict port of `rpython/memory/gctransform/shadowcolor.py`.
//!
//! The pass runs on the original flowspace graph, after
//! `gc_push_roots`/`gc_pop_roots` insertion and before backend lowering.  Rust
//! uses `Rc<RefCell<_>>` identity wrappers where upstream uses Python object
//! identity; the graph algorithm and its phase ordering are otherwise kept at
//! the upstream symbols named below.

#![expect(
    clippy::mutable_key_type,
    reason = "RPython keys these tables by Variable/Constant object identity; their Rust hashes exclude the mutable annotation and concretetype cells"
)]

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::flowspace::model::{
    Block, BlockKey, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, Hlvalue, Link,
    LinkKey, LinkRef, SpaceOperation, Variable, checkgraph, mkentrymap,
};
use crate::tool::algo::regalloc::{FlowRegAllocator, perform_flowspace_register_allocation};
use crate::tool::algo::unionfind::UnionFind;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::simplify::join_blocks;
use crate::translator::unsimplify::{
    insert_empty_block, insert_empty_startblock, split_block, varoftype,
};

type VarSet = FxHashSet<Variable>;
type VarMap<T> = FxHashMap<Variable, T>;

/// `shadowcolor.py::GCBitmaskTooLong`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GCBitmaskTooLong {
    graph: String,
}

impl fmt::Display for GCBitmaskTooLong {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the graph {:?} is too complex: cannot create a bitmask telling than more than 31/63 shadowstack entries are unused",
            self.graph
        )
    }
}

impl std::error::Error for GCBitmaskTooLong {}

/// `shadowcolor.py::PostProcessCheckError`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PostProcessCheckError(pub String);

impl fmt::Display for PostProcessCheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PostProcessCheckError {}

/// `shadowcolor.py::is_trivial_rewrite`.
pub fn is_trivial_rewrite(op: &SpaceOperation) -> bool {
    matches!(
        op.opname.as_str(),
        "same_as" | "cast_pointer" | "cast_opaque_ptr"
    ) && matches!(op.args.first(), Some(Hlvalue::Variable(_)))
}

/// `shadowcolor.py::find_predecessors`.
pub fn find_predecessors(
    graph: &mut FunctionGraph,
    mut pending_pred: Vec<(BlockRef, Variable)>,
) -> VarSet {
    let mut entrymap = mkentrymap(graph);
    if entrymap
        .get(&BlockKey::of(&graph.startblock))
        .map_or(0, Vec::len)
        != 1
    {
        insert_empty_startblock(graph);
        entrymap = mkentrymap(graph);
    }

    let mut pred: VarSet = pending_pred.iter().map(|(_, var)| var.clone()).collect();
    while let Some((block, var)) = pending_pred.pop() {
        let input_index = block
            .borrow()
            .inputargs
            .iter()
            .position(|value| value == &Hlvalue::Variable(var.clone()));
        if let Some(var_index) = input_index {
            for link in entrymap.get(&BlockKey::of(&block)).into_iter().flatten() {
                let link = link.borrow();
                let Some(prevblock) = link.prevblock.as_ref().and_then(|weak| weak.upgrade())
                else {
                    continue;
                };
                if let Some(Some(Hlvalue::Variable(source))) = link.args.get(var_index)
                    && pred.insert(source.clone())
                {
                    pending_pred.push((prevblock, source.clone()));
                }
            }
        } else {
            let operations = block.borrow().operations.clone();
            for op in operations {
                if op.result == Hlvalue::Variable(var.clone()) {
                    if is_trivial_rewrite(&op)
                        && let Hlvalue::Variable(source) = &op.args[0]
                        && pred.insert(source.clone())
                    {
                        pending_pred.push((block.clone(), source.clone()));
                    }
                    break;
                }
            }
        }
    }
    pred
}

/// `shadowcolor.py::find_successors`.
pub fn find_successors(
    _graph: &FunctionGraph,
    mut pending_succ: Vec<(BlockRef, Variable)>,
) -> VarSet {
    let mut succ: VarSet = pending_succ.iter().map(|(_, var)| var.clone()).collect();
    while let Some((block, var)) = pending_succ.pop() {
        let (operations, exits) = {
            let block = block.borrow();
            (block.operations.clone(), block.exits.clone())
        };
        for op in operations {
            if op.args.first() == Some(&Hlvalue::Variable(var.clone()))
                && is_trivial_rewrite(&op)
                && let Hlvalue::Variable(result) = &op.result
                && succ.insert(result.clone())
            {
                pending_succ.push((block.clone(), result.clone()));
            }
        }
        for link in exits {
            let (args, target) = {
                let link = link.borrow();
                (link.args.clone(), link.target.clone().expect("link.target"))
            };
            for (index, value) in args.iter().enumerate() {
                if value.as_ref() == Some(&Hlvalue::Variable(var.clone()))
                    && let Hlvalue::Variable(target_var) = &target.borrow().inputargs[index]
                    && succ.insert(target_var.clone())
                {
                    pending_succ.push((target.clone(), target_var.clone()));
                }
            }
        }
    }
    succ
}

/// `shadowcolor.py::find_interesting_variables`.
pub fn find_interesting_variables(graph: &mut FunctionGraph) -> Option<VarSet> {
    let mut pending_pred = Vec::new();
    let mut pending_succ = Vec::new();
    let mut interesting_vars = VarSet::default();
    for block in graph.iterblocks() {
        for op in &block.borrow().operations {
            if op.opname == "gc_push_roots" {
                for value in &op.args {
                    if let Hlvalue::Variable(var) = value {
                        interesting_vars.insert(var.clone());
                        pending_pred.push((block.clone(), var.clone()));
                    }
                }
            } else if op.opname == "gc_pop_roots" {
                for value in &op.args {
                    if let Hlvalue::Variable(var) = value {
                        assert!(
                            interesting_vars.contains(var),
                            "root must be pushed just above"
                        );
                        pending_succ.push((block.clone(), var.clone()));
                    }
                }
            }
        }
    }
    if interesting_vars.is_empty() {
        return None;
    }
    let pred = find_predecessors(graph, pending_pred);
    let succ = find_successors(graph, pending_succ);
    interesting_vars.extend(pred.intersection(&succ).cloned());
    Some(interesting_vars)
}

/// `shadowcolor.py::allocate_registers`.
pub fn allocate_registers(graph: &mut FunctionGraph) -> Option<FlowRegAllocator> {
    let interesting_vars = find_interesting_variables(graph)?;
    Some(perform_flowspace_register_allocation(graph, &|var| {
        interesting_vars.contains(var)
    }))
}

fn signed_constant(value: i64) -> Hlvalue {
    Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::Int(value),
        LowLevelType::Signed,
    ))
}

/// `shadowcolor.py::_gc_save_root`.
fn gc_save_root(index: usize, value: Hlvalue) -> SpaceOperation {
    SpaceOperation::new(
        "gc_save_root",
        vec![signed_constant(index as i64), value],
        Hlvalue::Variable(varoftype(LowLevelType::Void, None)),
    )
}

/// `shadowcolor.py::_gc_restore_root`.
fn gc_restore_root(index: usize, value: Hlvalue) -> SpaceOperation {
    SpaceOperation::new(
        "gc_restore_root",
        vec![signed_constant(index as i64), value],
        Hlvalue::Variable(varoftype(LowLevelType::Void, None)),
    )
}

/// `shadowcolor.py::make_bitmask`.
pub fn make_bitmask(
    filled: &[bool],
    graph: impl Into<String>,
) -> Result<(Option<usize>, Option<i64>), GCBitmaskTooLong> {
    let graph = graph.into();
    if filled.iter().all(|filled| *filled) {
        return Ok((None, None));
    }
    let mut bitmask: u128 = 0;
    let mut last_index = 0;
    for (index, is_filled) in filled.iter().enumerate() {
        if !is_filled {
            let shift = index - last_index;
            if bitmask != 0
                && (shift >= isize::BITS as usize || bitmask > (isize::MAX as u128 >> shift))
            {
                return Err(GCBitmaskTooLong { graph });
            }
            bitmask <<= shift;
            last_index = index;
            bitmask |= 1;
        }
    }
    assert_eq!(bitmask & 1, 1);
    if bitmask > isize::MAX as u128 {
        return Err(GCBitmaskTooLong { graph });
    }
    let bitmask = bitmask as i64;
    assert!(bitmask > 0);
    Ok((Some(last_index), Some(bitmask)))
}

/// `shadowcolor.py::expand_one_push_roots`.
pub fn expand_one_push_roots(
    regalloc: Option<&FlowRegAllocator>,
    args: &[Variable],
    graph: &str,
) -> Result<Vec<SpaceOperation>, GCBitmaskTooLong> {
    let Some(regalloc) = regalloc else {
        assert!(args.is_empty());
        return Ok(Vec::new());
    };
    let mut filled = vec![false; regalloc.numcolors];
    let mut result = Vec::new();
    for var in args {
        let index = regalloc.getcolor(var);
        assert!(!filled[index]);
        filled[index] = true;
        result.push(gc_save_root(index, Hlvalue::Variable(var.clone())));
    }
    let (bitmask_index, bitmask) = make_bitmask(&filled, graph)?;
    if let (Some(bitmask_index), Some(bitmask)) = (bitmask_index, bitmask) {
        result.push(gc_save_root(bitmask_index, signed_constant(bitmask)));
    }
    Ok(result)
}

/// `shadowcolor.py::expand_one_pop_roots`.
pub fn expand_one_pop_roots(
    regalloc: Option<&FlowRegAllocator>,
    args: &[Variable],
) -> Vec<SpaceOperation> {
    let Some(regalloc) = regalloc else {
        assert!(args.is_empty());
        return Vec::new();
    };
    args.iter()
        .map(|var| gc_restore_root(regalloc.getcolor(var), Hlvalue::Variable(var.clone())))
        .collect()
}

/// `shadowcolor.py::expand_push_roots`.
pub fn expand_push_roots(
    graph: &FunctionGraph,
    regalloc: Option<&FlowRegAllocator>,
) -> Result<(), GCBitmaskTooLong> {
    for block in graph.iterblocks() {
        let operations = block.borrow().operations.clone();
        let mut any_change = false;
        let mut newops = Vec::new();
        for op in operations {
            if op.opname == "gc_push_roots" {
                let args: Vec<_> = op
                    .args
                    .iter()
                    .filter_map(|value| match value {
                        Hlvalue::Variable(var) => Some(var.clone()),
                        _ => None,
                    })
                    .collect();
                newops.extend(expand_one_push_roots(regalloc, &args, &graph.name)?);
                any_change = true;
            } else {
                newops.push(op);
            }
        }
        if any_change {
            block.borrow_mut().operations = newops;
        }
    }
    Ok(())
}

fn constant_int(value: &Hlvalue) -> Option<i64> {
    match value {
        Hlvalue::Constant(Constant {
            value: ConstValue::Int(value),
            ..
        }) => Some(*value),
        _ => None,
    }
}

fn is_signed_constant(value: &Hlvalue) -> bool {
    matches!(
        value,
        Hlvalue::Constant(Constant {
            concretetype: Some(LowLevelType::Signed),
            ..
        })
    )
}

#[derive(Clone)]
struct SaveSite {
    block_key: BlockKey,
    block: BlockRef,
    operation: SpaceOperation,
}

impl PartialEq for SaveSite {
    fn eq(&self, other: &Self) -> bool {
        self.block_key == other.block_key && self.operation == other.operation
    }
}

impl Eq for SaveSite {}

impl std::hash::Hash for SaveSite {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.block_key, state);
        std::hash::Hash::hash(&self.operation, state);
    }
}

struct PushPart {
    index: usize,
    variables: VarSet,
    save_sites: HashSet<SaveSite>,
}

/// `shadowcolor.py::move_pushes_earlier`.
pub fn move_pushes_earlier(graph: &FunctionGraph, regalloc: Option<&FlowRegAllocator>) {
    let Some(regalloc) = regalloc else {
        return;
    };

    let entrymap = mkentrymap(graph);
    assert_eq!(entrymap[&BlockKey::of(&graph.startblock)].len(), 1);

    let mut inputvars: VarMap<(BlockRef, usize)> = VarMap::default();
    for block in graph.iterblocks() {
        for (index, value) in block.borrow().inputargs.iter().enumerate() {
            if let Hlvalue::Variable(var) = value {
                inputvars.insert(var.clone(), (block.clone(), index));
            }
        }
    }

    let mut parts = Vec::new();
    for index in 0..regalloc.numcolors {
        let mut unionfind = UnionFind::new(|_: &Variable| ());
        let mut successors = VarSet::default();

        for block in graph.iterblocks() {
            let operations = block.borrow().operations.clone();
            let Some(pop) = operations
                .iter()
                .rev()
                .find(|operation| operation.opname == "gc_pop_roots")
            else {
                continue;
            };
            let Some(var) = pop.args.iter().find_map(|value| match value {
                Hlvalue::Variable(var) if regalloc.checkcolor(var, index) => Some(var.clone()),
                _ => None,
            }) else {
                continue;
            };

            let mut succ = VarSet::default();
            let mut pending_succ = vec![(block.clone(), var)];
            while let Some((block1, var1)) = pending_succ.pop() {
                assert!(regalloc.checkcolor(&var1, index));
                let (operations, exits) = {
                    let block1 = block1.borrow();
                    (block1.operations.clone(), block1.exits.clone())
                };
                for operation in operations {
                    if is_trivial_rewrite(&operation)
                        && operation.args[0] == Hlvalue::Variable(var1.clone())
                        && let Hlvalue::Variable(result) = operation.result
                        && regalloc.checkcolor(&result, index)
                    {
                        pending_succ.push((block1.clone(), result));
                    }
                }
                for link in exits {
                    let (args, target) = {
                        let link = link.borrow();
                        (link.args.clone(), link.target.clone().expect("link.target"))
                    };
                    for (arg_index, value) in args.iter().enumerate() {
                        if value.as_ref() != Some(&Hlvalue::Variable(var1.clone())) {
                            continue;
                        }
                        let Hlvalue::Variable(target_var) =
                            target.borrow().inputargs[arg_index].clone()
                        else {
                            continue;
                        };
                        if succ.contains(&target_var) || !regalloc.checkcolor(&target_var, index) {
                            continue;
                        }
                        succ.insert(target_var.clone());
                        let has_barrier = target.borrow().operations.iter().any(|operation| {
                            matches!(operation.opname.as_str(), "gc_save_root" | "gc_pop_roots")
                        });
                        if !has_barrier {
                            pending_succ.push((target.clone(), target_var));
                        }
                    }
                }
            }
            let succ_list: Vec<_> = succ.iter().cloned().collect();
            unionfind.union_list(&succ_list);
            successors.extend(succ);
        }

        let mut save_sites_by_pred: VarMap<HashSet<SaveSite>> = VarMap::default();
        for block in graph.iterblocks() {
            let operations = block.borrow().operations.clone();
            let mut found = None;
            for (opindex, operation) in operations.iter().enumerate() {
                if operation.opname != "gc_save_root" {
                    continue;
                }
                if is_signed_constant(&operation.args[1]) {
                    break;
                }
                if constant_int(&operation.args[0]) == Some(index as i64) {
                    found = Some((opindex, operation.clone()));
                    break;
                }
            }
            let Some((opindex, operation)) = found else {
                continue;
            };
            let Hlvalue::Variable(saved_var) = operation.args[1].clone() else {
                continue;
            };
            let site = SaveSite {
                block_key: BlockKey::of(&block),
                block: block.clone(),
                operation,
            };
            let mut pred = VarSet::default();
            let mut pending_pred = vec![(block.clone(), saved_var, opindex)];
            while let Some((block1, mut var1, opindex1)) = pending_pred.pop() {
                assert_eq!(regalloc.getcolor(&var1), index);
                let operations = block1.borrow().operations.clone();
                let mut reached_input = true;
                for operation in operations[..opindex1].iter().rev() {
                    if operation.opname == "gc_pop_roots" {
                        reached_input = false;
                        break;
                    }
                    if operation.result == Hlvalue::Variable(var1.clone()) {
                        if !is_trivial_rewrite(operation) {
                            reached_input = false;
                            break;
                        }
                        let Hlvalue::Variable(source) = &operation.args[0] else {
                            reached_input = false;
                            break;
                        };
                        if !regalloc.checkcolor(source, index) {
                            reached_input = false;
                            break;
                        }
                        var1 = source.clone();
                    }
                }
                if !reached_input {
                    continue;
                }
                let varindex = block1
                    .borrow()
                    .inputargs
                    .iter()
                    .position(|value| value == &Hlvalue::Variable(var1.clone()))
                    .expect("predecessor variable is a block input");
                if !pred.insert(var1.clone()) {
                    continue;
                }
                for link in entrymap.get(&BlockKey::of(&block1)).into_iter().flatten() {
                    let link = link.borrow();
                    let Some(prevblock) = link.prevblock.as_ref().and_then(|weak| weak.upgrade())
                    else {
                        continue;
                    };
                    if let Some(Some(Hlvalue::Variable(source))) = link.args.get(varindex)
                        && !pred.contains(source)
                        && regalloc.checkcolor(source, index)
                    {
                        pending_pred.push((
                            prevblock.clone(),
                            source.clone(),
                            prevblock.borrow().operations.len(),
                        ));
                    }
                }
            }
            let pred_list: Vec<_> = pred.iter().cloned().collect();
            unionfind.union_list(&pred_list);
            for var in pred {
                save_sites_by_pred
                    .entry(var)
                    .or_default()
                    .insert(site.clone());
            }
        }

        let matching: Vec<_> = successors
            .intersection(&save_sites_by_pred.keys().cloned().collect())
            .cloned()
            .collect();
        let mut part_index_by_rep: VarMap<usize> = VarMap::default();
        for var in matching {
            let representative = unionfind.find_rep(var.clone());
            let part_index = if let Some(part_index) = part_index_by_rep.get(&representative) {
                *part_index
            } else {
                let part_index = parts.len();
                parts.push(PushPart {
                    index,
                    variables: VarSet::default(),
                    save_sites: HashSet::new(),
                });
                part_index_by_rep.insert(representative, part_index);
                part_index
            };
            let part = &mut parts[part_index];
            part.variables.insert(var.clone());
            part.save_sites
                .extend(save_sites_by_pred[&var].iter().cloned());
        }
    }

    parts.sort_by(|left, right| {
        let left = left.variables.len() as f64 / left.save_sites.len() as f64;
        let right = right.variables.len() as f64 / right.save_sites.len() as f64;
        left.partial_cmp(&right).unwrap()
    });

    let mut variables_along_changes: VarMap<(BlockRef, usize)> = VarMap::default();
    let mut live_at_start_of_block: HashSet<(BlockKey, usize)> = HashSet::new();
    let mut insert_gc_push_root: HashMap<LinkKey, (LinkRef, Vec<(usize, Hlvalue)>)> =
        HashMap::new();

    for part in parts {
        if part.variables.iter().any(|var| {
            let block = &inputvars[var].0;
            live_at_start_of_block.contains(&(BlockKey::of(block), part.index))
        }) {
            continue;
        }
        if part
            .save_sites
            .iter()
            .any(|site| !site.block.borrow().operations.contains(&site.operation))
        {
            continue;
        }
        for var in &part.variables {
            assert_eq!(regalloc.getcolor(var), part.index);
            assert!(!variables_along_changes.contains_key(var));
        }

        let mut success_count = 0;
        let mut mark = Vec::new();
        for var in &part.variables {
            let (block, varindex) = &inputvars[var];
            for link in entrymap.get(&BlockKey::of(block)).into_iter().flatten() {
                let mut value = link.borrow().args[*varindex].clone();
                let prevoperations = link
                    .borrow()
                    .prevblock
                    .as_ref()
                    .and_then(|weak| weak.upgrade())
                    .map(|block| block.borrow().operations.clone())
                    .unwrap_or_default();
                let mut decided = false;
                for operation in prevoperations.iter().rev() {
                    if operation.opname == "gc_pop_roots" {
                        if let Some(Hlvalue::Variable(value_var)) = &value {
                            if operation
                                .args
                                .contains(&Hlvalue::Variable(value_var.clone()))
                                && regalloc.checkcolor(value_var, part.index)
                            {
                                success_count += 1;
                            } else {
                                mark.push((part.index, link.clone(), *varindex));
                            }
                        } else {
                            mark.push((part.index, link.clone(), *varindex));
                        }
                        decided = true;
                        break;
                    }
                    if value.as_ref() == Some(&operation.result) {
                        if is_trivial_rewrite(operation)
                            && let Hlvalue::Variable(source) = &operation.args[0]
                            && regalloc.checkcolor(source, part.index)
                        {
                            value = Some(Hlvalue::Variable(source.clone()));
                        } else {
                            mark.push((part.index, link.clone(), *varindex));
                            decided = true;
                            break;
                        }
                    }
                }
                if !decided {
                    match value {
                        Some(Hlvalue::Variable(ref value_var))
                            if part.variables.contains(value_var) => {}
                        _ => mark.push((part.index, link.clone(), *varindex)),
                    }
                }
            }
        }

        if success_count > 0 {
            for site in &part.save_sites {
                let mut block = site.block.borrow_mut();
                let position = block
                    .operations
                    .iter()
                    .position(|operation| operation == &site.operation)
                    .expect("gc_save_root disappeared");
                block.operations.remove(position);
            }
            for (index, link, varindex) in mark {
                let value = link.borrow().args[varindex]
                    .clone()
                    .expect("root link argument");
                insert_gc_push_root
                    .entry(LinkKey::of(&link))
                    .or_insert_with(|| (link.clone(), Vec::new()))
                    .1
                    .push((index, value));
            }
            for var in &part.variables {
                let block = &inputvars[var].0;
                variables_along_changes.insert(var.clone(), (block.clone(), part.index));
                live_at_start_of_block.insert((BlockKey::of(block), part.index));
            }
        }
    }

    for (_, (link, mut insertions)) in insert_gc_push_root {
        insertions.sort_by_key(|(index, _)| *index);
        let newops = insertions
            .into_iter()
            .map(|(index, value)| gc_save_root(index, value))
            .collect();
        insert_empty_block(&link, newops);
    }
}

/// `shadowcolor.py::expand_pop_roots`.
pub fn expand_pop_roots(graph: &FunctionGraph, regalloc: Option<&FlowRegAllocator>) {
    let mut drop: VarMap<i64> = VarMap::default();
    for block in graph.iterblocks() {
        let operations = block.borrow().operations.clone();
        let mut any_change = false;
        let mut newops = Vec::new();
        for operation in operations {
            if operation.opname == "gc_pop_roots" {
                let args: Vec<_> = operation
                    .args
                    .iter()
                    .filter_map(|value| match value {
                        Hlvalue::Variable(var) => Some(var.clone()),
                        _ => None,
                    })
                    .collect();
                let expanded = expand_one_pop_roots(regalloc, &args);
                drop.clear();
                for restore in &expanded {
                    if let Hlvalue::Variable(var) = &restore.args[1] {
                        drop.insert(var.clone(), constant_int(&restore.args[0]).unwrap());
                    }
                }
                newops.extend(expanded);
                any_change = true;
            } else if operation.opname == "gc_save_root"
                && matches!(
                    &operation.args[1],
                    Hlvalue::Variable(var)
                        if drop.get(var).copied() == constant_int(&operation.args[0])
                )
            {
                any_change = true;
            } else {
                newops.push(operation);
            }
        }
        if any_change {
            block.borrow_mut().operations = newops;
        }
    }
}

fn is_interesting_frame_op(operation: &SpaceOperation, bitmask_all_free: Option<i64>) -> bool {
    if operation.opname == "gc_restore_root" {
        return true;
    }
    operation.opname == "gc_save_root"
        && !(is_signed_constant(&operation.args[1])
            && constant_int(&operation.args[1]) == bitmask_all_free)
}

fn insert_along_link(
    link: &LinkRef,
    opname: &str,
    args: &[Hlvalue],
    cache: &mut HashMap<BlockKey, BlockRef>,
) {
    let target = link.borrow().target.clone().expect("link.target");
    let target_key = BlockKey::of(&target);
    let newblock = cache.entry(target_key).or_insert_with(|| {
        let inputargs = target
            .borrow()
            .inputargs
            .iter()
            .map(|value| match value {
                Hlvalue::Variable(var) => Hlvalue::Variable(var.copy()),
                Hlvalue::Constant(constant) => Hlvalue::Constant(constant.clone()),
            })
            .collect::<Vec<_>>();
        let newblock = Block::shared(inputargs.clone());
        newblock.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args.to_vec(),
            Hlvalue::Variable(varoftype(LowLevelType::Void, None)),
        ));
        newblock.closeblock(vec![
            Link::new(inputargs, Some(target.clone()), None).into_ref(),
        ]);
        newblock
    });
    link.borrow_mut().target = Some(newblock.clone());
}

/// `shadowcolor.py::add_enter_leave_roots_frame`.
pub fn add_enter_leave_roots_frame(
    graph: &mut FunctionGraph,
    regalloc: Option<&FlowRegAllocator>,
    c_gcdata: Hlvalue,
) {
    let Some(regalloc) = regalloc else {
        return;
    };

    for block in graph.iterblocks() {
        let last_restore = block
            .borrow()
            .operations
            .iter()
            .rposition(|operation| operation.opname == "gc_restore_root");
        if let Some(index) = last_restore
            && index + 1 < block.borrow().operations.len()
        {
            split_block(&block, index + 1, None);
        }
    }

    insert_empty_startblock(graph);
    let entrymap = mkentrymap(graph);
    let bitmask_all_free = 1_u64
        .checked_shl(regalloc.numcolors as u32)
        .and_then(|value| i64::try_from(value - 1).ok());

    let mut interesting_blocks = Vec::new();
    for block in graph.iterblocks() {
        if block
            .borrow()
            .operations
            .iter()
            .any(|operation| is_interesting_frame_op(operation, bitmask_all_free))
        {
            assert_ne!(BlockKey::of(&block), BlockKey::of(&graph.startblock));
            assert_ne!(BlockKey::of(&block), BlockKey::of(&graph.returnblock));
            interesting_blocks.push(block);
        }
    }

    let mut before_blocks = HashSet::new();
    let mut pending = interesting_blocks.clone();
    let mut seen: HashSet<_> = interesting_blocks.iter().map(BlockKey::of).collect();
    while let Some(block) = pending.pop() {
        for link in &block.borrow().exits {
            let target = link.borrow().target.clone().expect("link.target");
            let target_key = BlockKey::of(&target);
            before_blocks.insert(target_key.clone());
            if seen.insert(target_key) {
                pending.push(target);
            }
        }
    }
    assert!(!before_blocks.contains(&BlockKey::of(&graph.startblock)));

    let mut after_blocks: HashSet<_> = interesting_blocks.iter().map(BlockKey::of).collect();
    let mut pending = interesting_blocks.clone();
    while let Some(block) = pending.pop() {
        for link in entrymap.get(&BlockKey::of(&block)).into_iter().flatten() {
            if let Some(prevblock) = link
                .borrow()
                .prevblock
                .as_ref()
                .and_then(|weak| weak.upgrade())
            {
                let prev_key = BlockKey::of(&prevblock);
                if after_blocks.insert(prev_key) {
                    pending.push(prevblock);
                }
            }
        }
    }
    assert!(!after_blocks.contains(&BlockKey::of(&graph.returnblock)));

    let inside_blocks: HashSet<_> = before_blocks.intersection(&after_blocks).cloned().collect();
    let mut inside_or_interesting_blocks = inside_blocks.clone();
    inside_or_interesting_blocks.extend(interesting_blocks.iter().map(BlockKey::of));

    let c_num = signed_constant(regalloc.numcolors as i64);
    for block in &interesting_blocks {
        if !inside_blocks.contains(&BlockKey::of(block)) {
            let index = block
                .borrow()
                .operations
                .iter()
                .position(|operation| is_interesting_frame_op(operation, bitmask_all_free))
                .unwrap();
            block.borrow_mut().operations.insert(
                index,
                SpaceOperation::new(
                    "gc_enter_roots_frame",
                    vec![c_gcdata.clone(), c_num.clone()],
                    Hlvalue::Variable(varoftype(LowLevelType::Void, None)),
                ),
            );
        }
    }

    let mut enter_cache = HashMap::new();
    let mut leave_cache = HashMap::new();
    for block in graph.iterblocks() {
        let block_key = BlockKey::of(&block);
        let exits = block.borrow().exits.clone();
        if !inside_or_interesting_blocks.contains(&block_key) {
            for link in exits {
                let target = link.borrow().target.clone().expect("link.target");
                if inside_blocks.contains(&BlockKey::of(&target)) {
                    insert_along_link(
                        &link,
                        "gc_enter_roots_frame",
                        &[c_gcdata.clone(), c_num.clone()],
                        &mut enter_cache,
                    );
                }
            }
        } else {
            for link in exits {
                let target = link.borrow().target.clone().expect("link.target");
                if !inside_blocks.contains(&BlockKey::of(&target)) {
                    insert_along_link(&link, "gc_leave_roots_frame", &[], &mut leave_cache);
                }
            }
        }
    }

    for block in graph.iterblocks() {
        if inside_blocks.contains(&BlockKey::of(&block)) {
            continue;
        }
        let operations = block.borrow().operations.clone();
        let mut newops = Vec::new();
        for (index, operation) in operations.iter().enumerate() {
            if operation.opname == "gc_enter_roots_frame" {
                newops.extend_from_slice(&operations[index..]);
                break;
            }
            if operation.opname != "gc_save_root"
                || is_interesting_frame_op(operation, bitmask_all_free)
            {
                newops.push(operation.clone());
            }
        }
        if newops.len() < operations.len() {
            block.borrow_mut().operations = newops;
        }
    }

    join_blocks(graph);
}

fn check_error(graph: &FunctionGraph, detail: impl fmt::Display) -> PostProcessCheckError {
    PostProcessCheckError(format!("{}: {}", graph.name, detail))
}

/// `shadowcolor.py::postprocess_double_check`.
pub fn postprocess_double_check(graph: &FunctionGraph) -> Result<(), PostProcessCheckError> {
    let mut saved: HashMap<Hlvalue, HashSet<i64>> = HashMap::new();
    let mut in_frame: HashMap<BlockKey, bool> = HashMap::new();
    let start_key = BlockKey::of(&graph.startblock);
    in_frame.insert(start_key.clone(), false);
    let mut pending: HashMap<BlockKey, BlockRef> =
        HashMap::from([(start_key, graph.startblock.clone())]);

    while let Some(key) = pending.keys().next().cloned() {
        let block = pending.remove(&key).unwrap();
        let (inputargs, operations, exits) = {
            let block = block.borrow();
            (
                block.inputargs.clone(),
                block.operations.clone(),
                block.exits.clone(),
            )
        };
        let mut locsaved: HashMap<Hlvalue, HashSet<i64>> = HashMap::new();
        let mut currently_in_frame = in_frame[&key];
        if currently_in_frame {
            for value in &inputargs {
                locsaved.insert(value.clone(), saved[value].clone());
            }
        }

        for operation in operations {
            match operation.opname.as_str() {
                "gc_restore_root" => {
                    if !currently_in_frame {
                        return Err(check_error(graph, "gc_restore_root: no frame!"));
                    }
                    if matches!(operation.args[1], Hlvalue::Constant(_)) {
                        continue;
                    }
                    let num = constant_int(&operation.args[0]).unwrap();
                    if !locsaved
                        .get(&operation.args[1])
                        .is_some_and(|locations| locations.contains(&num))
                    {
                        return Err(check_error(
                            graph,
                            format!("gc_restore_root {num}: root is not saved"),
                        ));
                    }
                }
                "gc_save_root" => {
                    if !currently_in_frame {
                        return Err(check_error(graph, "gc_save_root: no frame!"));
                    }
                    let num = constant_int(&operation.args[0]).unwrap();
                    for locations in locsaved.values_mut() {
                        locations.remove(&num);
                    }
                    match &operation.args[1] {
                        Hlvalue::Variable(_) => {
                            locsaved
                                .entry(operation.args[1].clone())
                                .or_default()
                                .insert(num);
                        }
                        Hlvalue::Constant(_) if !is_signed_constant(&operation.args[1]) => {
                            locsaved
                                .entry(operation.args[1].clone())
                                .or_default()
                                .insert(num);
                        }
                        Hlvalue::Constant(_) => {
                            let bitmask = constant_int(&operation.args[1]).unwrap();
                            if bitmask != 1 {
                                assert_eq!(bitmask & 1, 1);
                                assert!(1 < bitmask && bitmask < (2_i64 << num));
                                let nummask: Vec<_> = (0..=num)
                                    .filter(|index| bitmask & (1_i64 << (num - index)) != 0)
                                    .collect();
                                assert_eq!(nummask.last(), Some(&num));
                                for locations in locsaved.values_mut() {
                                    locations.retain(|index| !nummask.contains(index));
                                }
                            }
                        }
                    }
                }
                "gc_enter_roots_frame" => {
                    if currently_in_frame {
                        return Err(check_error(graph, "double enter"));
                    }
                    currently_in_frame = true;
                    for value in &inputargs {
                        locsaved.insert(value.clone(), HashSet::new());
                    }
                }
                "gc_leave_roots_frame" => {
                    if !currently_in_frame {
                        return Err(check_error(graph, "not entered"));
                    }
                    currently_in_frame = false;
                }
                _ if is_trivial_rewrite(&operation) && currently_in_frame => {
                    let locations = locsaved[&operation.args[0]].clone();
                    locsaved.insert(operation.result, locations);
                }
                _ => {
                    locsaved.insert(operation.result, HashSet::new());
                }
            }
        }

        for link in exits {
            let (args, target) = {
                let link = link.borrow();
                (link.args.clone(), link.target.clone().expect("link.target"))
            };
            let target_key = BlockKey::of(&target);
            let mut changed = false;
            match in_frame.get(&target_key) {
                None => {
                    in_frame.insert(target_key.clone(), currently_in_frame);
                    changed = true;
                }
                Some(value) if *value != currently_in_frame => {
                    return Err(check_error(graph, "inconsistent in_frame"));
                }
                _ => {}
            }
            if currently_in_frame {
                for (index, value) in args.iter().enumerate() {
                    let value = value.as_ref().expect("complete graph link argument");
                    let mut locations = locsaved.get(value).cloned().unwrap_or_else(|| {
                        assert!(matches!(value, Hlvalue::Constant(_)));
                        HashSet::new()
                    });
                    let target_value = target.borrow().inputargs[index].clone();
                    if let Some(previous) = saved.get(&target_value) {
                        if locations == *previous {
                            continue;
                        }
                        locations = locations.intersection(previous).copied().collect();
                    }
                    saved.insert(target_value, locations);
                    changed = true;
                }
            }
            if changed {
                pending.insert(target_key, target);
            }
        }
    }

    if in_frame
        .get(&BlockKey::of(&graph.returnblock))
        .copied()
        .unwrap_or(false)
    {
        return Err(check_error(graph, "missing gc_leave_roots_frame"));
    }
    assert!(!saved.contains_key(&graph.getreturnvar()));
    Ok(())
}

/// `shadowcolor.py::postprocess_graph`.
pub fn postprocess_graph(
    graph: &mut FunctionGraph,
    c_gcdata: Hlvalue,
) -> Result<bool, Box<dyn std::error::Error>> {
    let regalloc = allocate_registers(graph);
    expand_push_roots(graph, regalloc.as_ref())?;
    move_pushes_earlier(graph, regalloc.as_ref());
    expand_pop_roots(graph, regalloc.as_ref());
    add_enter_leave_roots_frame(graph, regalloc.as_ref(), c_gcdata);
    checkgraph(graph);
    postprocess_double_check(graph)?;
    Ok(regalloc.is_some())
}

/// `shadowcolor.py::postprocess_inlining`.
pub fn postprocess_inlining(graph: &FunctionGraph) -> Result<(), PostProcessCheckError> {
    for block in graph.iterblocks() {
        let operations = block.borrow().operations.clone();
        for index in (0..operations.len()).rev() {
            if operations[index].opname == "gc_pop_roots" {
                break;
            }
            if operations[index].opname == "gc_push_roots" {
                fix_graph_after_inlining(graph, block.clone(), index)?;
                break;
            }
        }
    }
    checkgraph(graph);
    Ok(())
}

/// `shadowcolor.py::_fix_graph_after_inlining`.
fn fix_graph_after_inlining(
    graph: &FunctionGraph,
    initial_block: BlockRef,
    initial_index: usize,
) -> Result<(), PostProcessCheckError> {
    let operation = initial_block.borrow_mut().operations.remove(initial_index);
    assert_eq!(operation.opname, "gc_push_roots");
    let mut seen = HashSet::new();
    let mut pending = vec![(initial_block, initial_index, operation.args)];
    while let Some((block, start_index, track_args)) = pending.pop() {
        if !seen.insert(BlockKey::of(&block)) {
            continue;
        }
        assert!(!block.borrow().is_final_block());
        let operations = block.borrow().operations.clone();
        let mut new_operations = operations[..start_index].to_vec();
        let mut stop = false;
        for (index, operation) in operations.iter().enumerate().skip(start_index) {
            if operation.opname == "gc_push_roots" {
                return Err(check_error(
                    graph,
                    "seems to have inlined inside it another graph which also uses GC roots",
                ));
            }
            if operation.opname == "gc_pop_roots" {
                new_operations.extend_from_slice(&operations[index + 1..]);
                stop = true;
                break;
            }
            if matches!(operation.opname.as_str(), "direct_call" | "indirect_call") {
                new_operations.push(SpaceOperation::new(
                    "gc_push_roots",
                    track_args.clone(),
                    Hlvalue::Variable(varoftype(LowLevelType::Void, None)),
                ));
                new_operations.push(operation.clone());
                new_operations.push(SpaceOperation::new(
                    "gc_pop_roots",
                    track_args.clone(),
                    Hlvalue::Variable(varoftype(LowLevelType::Void, None)),
                ));
            } else {
                new_operations.push(operation.clone());
            }
        }
        block.borrow_mut().operations = new_operations;
        if !stop {
            for link in block.borrow().exits.clone() {
                let (args, target) = {
                    let link = link.borrow();
                    (link.args.clone(), link.target.clone().expect("link.target"))
                };
                let mut track_next = Vec::new();
                for value in &track_args {
                    let Hlvalue::Variable(_) = value else {
                        continue;
                    };
                    let index = args
                        .iter()
                        .position(|arg| arg.as_ref() == Some(value))
                        .expect("tracked root must be on link");
                    track_next.push(target.borrow().inputargs[index].clone());
                }
                pending.push((target, 0, track_next));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn void_result() -> Hlvalue {
        Hlvalue::Variable(varoftype(LowLevelType::Void, None))
    }

    fn marker(opname: &str, args: Vec<Hlvalue>) -> SpaceOperation {
        SpaceOperation::new(opname, args, void_result())
    }

    fn direct_call() -> SpaceOperation {
        marker(
            "direct_call",
            vec![Hlvalue::Constant(Constant::new(ConstValue::Int(0)))],
        )
    }

    fn linear_graph(name: &str, input: Variable, operations: Vec<SpaceOperation>) -> FunctionGraph {
        let start = Block::shared(vec![Hlvalue::Variable(input.clone())]);
        start.borrow_mut().operations = operations;
        let graph = FunctionGraph::new(name, start.clone());
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(input)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        graph
    }

    fn opnames(graph: &FunctionGraph) -> Vec<String> {
        graph
            .iterblocks()
            .into_iter()
            .flat_map(|block| block.borrow().operations.clone())
            .map(|operation| operation.opname)
            .collect()
    }

    #[test]
    fn trivial_rewrite_matches_shadowcolor() {
        let source = Variable::named("source");
        let result = Variable::named("result");
        assert!(is_trivial_rewrite(&SpaceOperation::new(
            "same_as",
            vec![Hlvalue::Variable(source.clone())],
            Hlvalue::Variable(result.clone()),
        )));
        assert!(is_trivial_rewrite(&SpaceOperation::new(
            "cast_pointer",
            vec![Hlvalue::Variable(source.clone())],
            Hlvalue::Variable(result.clone()),
        )));
        assert!(!is_trivial_rewrite(&SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(source)],
            Hlvalue::Variable(result),
        )));
    }

    #[test]
    fn predecessor_and_successor_follow_rewrites_and_links() {
        let source = Variable::named("source");
        let rewritten = Variable::named("rewritten");
        let target_input = Variable::named("target");
        let start = Block::shared(vec![Hlvalue::Variable(source.clone())]);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "same_as",
            vec![Hlvalue::Variable(source.clone())],
            Hlvalue::Variable(rewritten.clone()),
        ));
        let target = Block::shared(vec![Hlvalue::Variable(target_input.clone())]);
        let graph = FunctionGraph::new("copies", start.clone());
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(rewritten.clone())],
                Some(target.clone()),
                None,
            )
            .into_ref(),
        ]);
        target.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(target_input.clone())],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let succ = find_successors(&graph, vec![(start, source.clone())]);
        assert!(succ.contains(&source));
        assert!(succ.contains(&rewritten));
        assert!(succ.contains(&target_input));

        let return_var = match graph.getreturnvar() {
            Hlvalue::Variable(var) => var,
            _ => unreachable!(),
        };
        let mut graph = graph;
        let returnblock = graph.returnblock.clone();
        let pred = find_predecessors(&mut graph, vec![(returnblock, return_var.clone())]);
        assert!(pred.contains(&return_var));
        assert!(pred.contains(&target_input));
        assert!(pred.contains(&rewritten));
        assert!(pred.contains(&source));
    }

    #[test]
    fn make_bitmask_matches_shadowcolor_encoding() {
        for bits in 0_u16..256 {
            let mut filled: Vec<_> = (0..8).map(|index| bits & (1 << index) != 0).collect();
            let (index, bitmask) = make_bitmask(&filled, "test").unwrap();
            match (index, bitmask) {
                (None, None) => {}
                (Some(mut index), Some(mut bitmask)) => {
                    while bitmask != 0 {
                        if bitmask & 1 != 0 {
                            assert!(!filled[index]);
                            filled[index] = true;
                        }
                        bitmask >>= 1;
                        if index == 0 {
                            break;
                        }
                        index -= 1;
                    }
                }
                _ => panic!("index and mask must be both present or absent"),
            }
            assert!(filled.iter().all(|value| *value));
        }
        let mut too_long = vec![true; 65];
        too_long[0] = false;
        too_long[64] = false;
        assert!(make_bitmask(&too_long, "large").is_err());
    }

    #[test]
    fn postprocess_graph_expands_and_checks_linear_root_frame() {
        let root = Variable::named("root");
        root.set_concretetype(Some(LowLevelType::Signed));
        let root_value = Hlvalue::Variable(root.clone());
        let operations = vec![
            marker("gc_push_roots", vec![root_value.clone()]),
            direct_call(),
            marker("gc_pop_roots", vec![root_value]),
        ];
        let mut graph = linear_graph("linear", root, operations);
        assert!(postprocess_graph(&mut graph, Constant::new(ConstValue::Int(0)).into()).unwrap());
        assert_eq!(
            opnames(&graph),
            [
                "gc_enter_roots_frame",
                "gc_save_root",
                "direct_call",
                "gc_restore_root",
                "gc_leave_roots_frame",
            ]
        );
        postprocess_double_check(&graph).unwrap();
    }

    #[test]
    fn empty_root_markers_disappear_without_a_frame() {
        let input = Variable::named("input");
        let mut graph = linear_graph(
            "empty",
            input,
            vec![
                marker("gc_push_roots", vec![]),
                direct_call(),
                marker("gc_pop_roots", vec![]),
            ],
        );
        assert!(!postprocess_graph(&mut graph, Constant::new(ConstValue::Int(0)).into()).unwrap());
        assert_eq!(opnames(&graph), ["direct_call"]);
    }

    #[test]
    fn expand_pop_drops_immediate_redundant_save() {
        let root = Variable::named("root");
        let root_value = Hlvalue::Variable(root.clone());
        let mut graph = linear_graph(
            "drop-save",
            root,
            vec![
                marker("gc_push_roots", vec![root_value.clone()]),
                marker("gc_pop_roots", vec![root_value.clone()]),
                marker("gc_push_roots", vec![root_value.clone()]),
                marker("gc_pop_roots", vec![root_value]),
            ],
        );
        let regalloc = allocate_registers(&mut graph).unwrap();
        expand_push_roots(&graph, Some(&regalloc)).unwrap();
        expand_pop_roots(&graph, Some(&regalloc));
        let names = opnames(&graph);
        assert_eq!(
            names
                .iter()
                .filter(|name| name.as_str() == "gc_save_root")
                .count(),
            1
        );
        assert_eq!(
            names
                .iter()
                .filter(|name| name.as_str() == "gc_restore_root")
                .count(),
            2
        );
    }

    #[test]
    fn move_pushes_earlier_reuses_a_save_across_blocks() {
        let first_root = Variable::named("first_root");
        let first_value = Hlvalue::Variable(first_root.clone());
        let first = Block::shared(vec![first_value.clone()]);
        first.borrow_mut().operations = vec![
            marker("gc_push_roots", vec![first_value.clone()]),
            direct_call(),
            marker("gc_pop_roots", vec![first_value]),
        ];

        let middle_root = Variable::named("middle_root");
        let middle = Block::shared(vec![Hlvalue::Variable(middle_root.clone())]);
        let last_root = Variable::named("last_root");
        let last_value = Hlvalue::Variable(last_root.clone());
        let last = Block::shared(vec![last_value.clone()]);
        last.borrow_mut().operations = vec![
            marker("gc_push_roots", vec![last_value.clone()]),
            direct_call(),
            marker("gc_pop_roots", vec![last_value]),
        ];

        let mut graph = FunctionGraph::new("reuse-save", first.clone());
        first.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(first_root)],
                Some(middle.clone()),
                None,
            )
            .into_ref(),
        ]);
        middle.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(middle_root)],
                Some(last.clone()),
                None,
            )
            .into_ref(),
        ]);
        last.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(last_root)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let regalloc = allocate_registers(&mut graph).unwrap();
        expand_push_roots(&graph, Some(&regalloc)).unwrap();
        move_pushes_earlier(&graph, Some(&regalloc));
        expand_pop_roots(&graph, Some(&regalloc));
        let names = opnames(&graph);
        assert_eq!(
            names
                .iter()
                .filter(|name| name.as_str() == "gc_save_root")
                .count(),
            1
        );
        assert_eq!(
            names
                .iter()
                .filter(|name| name.as_str() == "gc_restore_root")
                .count(),
            2
        );
        add_enter_leave_roots_frame(
            &mut graph,
            Some(&regalloc),
            Constant::new(ConstValue::Int(0)).into(),
        );
        postprocess_double_check(&graph).unwrap();
    }

    #[test]
    fn postprocess_inlining_moves_markers_around_calls() {
        let root = Variable::named("root");
        let root_value = Hlvalue::Variable(root.clone());
        let start = Block::shared(vec![root_value.clone()]);
        start.borrow_mut().operations = vec![
            marker("gc_push_roots", vec![root_value]),
            marker("int_add", vec![]),
        ];
        let call_root = Variable::named("call_root");
        let call_block = Block::shared(vec![Hlvalue::Variable(call_root.clone())]);
        call_block.borrow_mut().operations = vec![direct_call()];
        let tail_root = Variable::named("tail_root");
        let tail_block = Block::shared(vec![Hlvalue::Variable(tail_root.clone())]);
        tail_block.borrow_mut().operations = vec![marker(
            "gc_pop_roots",
            vec![Hlvalue::Variable(tail_root.clone())],
        )];
        let graph = FunctionGraph::new("inlined", start.clone());
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(root)],
                Some(call_block.clone()),
                None,
            )
            .into_ref(),
        ]);
        call_block.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(call_root)],
                Some(tail_block.clone()),
                None,
            )
            .into_ref(),
        ]);
        tail_block.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(tail_root)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        postprocess_inlining(&graph).unwrap();
        assert_eq!(
            opnames(&graph),
            ["int_add", "gc_push_roots", "direct_call", "gc_pop_roots"]
        );
    }
}
