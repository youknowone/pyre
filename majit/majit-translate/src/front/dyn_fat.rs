//! Keep the metadata word of a fat `&dyn` / `Box<dyn>` across a call.
//!
//! `FunctionGraph::set_return` carries one value, and `jtransform.py`
//! rewrites `direct_call` to `inline_call_*` so the meta-interpreter
//! descends into the callee jitcode at runtime. A `&dyn` is two words
//! (data, vtable). The one-word return keeps the data pointer and drops
//! the vtable, so a later `method_*` read on `{vtable}` loads the slot
//! from the data pointer.
//!
//! RPython never hits that shape: the trace-time `inline_call_*` descent
//! sees the callee body, where both words are still field reads. This
//! pass does the same only for a fat-dyn return whose body is in this
//! LLBC: splice that body (`inline::splice_direct_call`), then retarget
//! each `method_*` read so its base is the metadata word (`FatLen`) of
//! the fat field. The data word stays the call's receiver. Other calls
//! stay residual. A cross-crate fat return has no body here and is left
//! unchanged.

use std::collections::{HashMap, HashSet};

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::TypeDeclKind;

use crate::flowspace::model::Variable;
use crate::front::mir::tyref_is_fat_dyn;
use crate::inline::splice_direct_call;
use crate::model::{
    BlockId, ConcreteType, FieldDescriptor, FunctionGraph, LinkArg, OpKind, SpaceOperation,
    ValueType, VecFieldPart,
};

/// Bodies larger than this stay calls. `w_dict_get_strategy` is a handful
/// of blocks; a bigger fat-dyn return is not spliced.
pub(crate) const MAX_SPLICED_BLOCKS: usize = 24;

thread_local! {
    static SPLICE_STACK: std::cell::RefCell<Vec<u64>> = const { std::cell::RefCell::new(Vec::new()) };
}

struct SpliceGuard(u64);

impl Drop for SpliceGuard {
    fn drop(&mut self) {
        SPLICE_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            if let Some(pos) = stack.iter().rposition(|id| *id == self.0) {
                stack.remove(pos);
            }
        });
    }
}

pub(crate) fn splice_in_progress(def_id: u64) -> bool {
    SPLICE_STACK.with(|stack| stack.borrow().contains(&def_id))
}

fn enter_splice(def_id: u64) -> Option<SpliceGuard> {
    if splice_in_progress(def_id) {
        return None;
    }
    SPLICE_STACK.with(|stack| stack.borrow_mut().push(def_id));
    Some(SpliceGuard(def_id))
}

struct DynCallSite {
    block: BlockId,
    op_index: usize,
    def_id: u64,
}

/// Splice recorded fat-dyn calls, then point `{vtable}` `method_*` reads
/// at the metadata word. `lower_callee` returns the callee graph already
/// run through this pass, or `None` when the body must stay a call.
pub(crate) fn splice_fat_dyn_returns(
    graph: &mut FunctionGraph,
    llbc: &Llbc,
    fun_id: u64,
    dyn_results: &[Variable],
    lower_callee: &mut dyn FnMut(u64) -> Option<FunctionGraph>,
) {
    // Most graphs never form a fat-dyn value. Skip the type-decl walk
    // unless this body recorded a fat return or already reads a vtable
    // slot (a same-function `method_*` whose base is still the data word).
    if dyn_results.is_empty() && !graph_has_vtable_method(graph) {
        return;
    }
    let Some(_guard) = enter_splice(fun_id) else {
        return;
    };
    let sites = fat_dyn_call_sites(graph, dyn_results);
    let mut prepared = Vec::new();
    for site in &sites {
        if splice_in_progress(site.def_id) {
            continue;
        }
        if let Some(callee) = lower_callee(site.def_id) {
            prepared.push((site.block, site.op_index, callee));
        }
    }
    let mut spliced = 0usize;
    for (block, op_index, callee) in prepared.into_iter().rev() {
        if splice_direct_call(graph, block, op_index, callee) {
            spliced += 1;
        }
    }
    let retargeted = retarget_vtable_method_bases(graph, llbc);
    if spliced > 0 || retargeted > 0 {
        crate::model::clear_unreachable_blocks(graph);
        crate::model::prune_dead_phis(graph);
    }
}

fn graph_has_vtable_method(graph: &FunctionGraph) -> bool {
    graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| match &op.kind {
            OpKind::FieldRead { field, .. } => is_vtable_method(field),
            _ => false,
        })
    })
}

fn fat_dyn_call_sites(graph: &FunctionGraph, dyn_results: &[Variable]) -> Vec<DynCallSite> {
    let ids: HashSet<u64> = dyn_results.iter().map(Variable::id).collect();
    let mut sites = Vec::new();
    for block in &graph.blocks {
        for (op_index, op) in block.operations.iter().enumerate() {
            let Some(result) = op.result.as_ref() else {
                continue;
            };
            if !ids.contains(&result.id()) {
                continue;
            }
            let OpKind::Call { target, .. } = &op.kind else {
                continue;
            };
            let Some(def_id) = target.fun_decl_id() else {
                continue;
            };
            sites.push(DynCallSite {
                block: block.id,
                op_index,
                def_id,
            });
        }
    }
    sites
}

fn cast_instance_operand(kind: &OpKind) -> Option<Variable> {
    crate::model::cast_instance_root(kind)?;
    let OpKind::Call { args, .. } = kind else {
        return None;
    };
    args.first().and_then(LinkArg::as_variable).cloned()
}

fn is_vtable_method(field: &FieldDescriptor) -> bool {
    field.name.starts_with("method_")
        && field
            .owner_root
            .as_deref()
            .is_some_and(|owner| owner == "{vtable}" || owner.ends_with("::{vtable}"))
}

fn owner_matches(decl_path: &str, owner_root: &str) -> bool {
    if decl_path == owner_root {
        return true;
    }
    let decl_leaf = decl_path.rsplit("::").next().unwrap_or(decl_path);
    if owner_root == decl_leaf || decl_path.ends_with(&format!("::{owner_root}")) {
        return true;
    }
    let owner_leaf = owner_root.rsplit("::").next().unwrap_or(owner_root);
    let owner_leaf = owner_leaf.split('<').next().unwrap_or(owner_leaf);
    owner_leaf == decl_leaf
}

fn field_is_fat_dyn(llbc: &Llbc, owner_root: &str, field_name: &str) -> bool {
    let mut saw = false;
    let mut all_fat = true;
    for decl in llbc.iter_type_decls() {
        if !owner_matches(&decl.item_meta.name_path(), owner_root) {
            continue;
        }
        let TypeDeclKind::Struct(fields) = &decl.kind else {
            continue;
        };
        let Some(field) = fields
            .iter()
            .find(|field| field.name.as_deref() == Some(field_name))
        else {
            continue;
        };
        saw = true;
        if !tyref_is_fat_dyn(&field.ty, llbc) {
            all_fat = false;
        }
    }
    saw && all_fat
}

struct FatField {
    block: BlockId,
    var_id: u64,
    base: Variable,
    field: FieldDescriptor,
    pure: bool,
}

struct SameAs {
    block: BlockId,
    operand: Variable,
}

struct MetaEnv {
    fat_fields: HashMap<u64, FatField>,
    copies: HashMap<u64, SameAs>,
    fat_len: HashMap<u64, Variable>,
    memo: HashMap<(usize, u64), Variable>,
    visiting: HashSet<(usize, u64)>,
}

fn index_producers(
    graph: &FunctionGraph,
    llbc: &Llbc,
) -> (HashMap<u64, FatField>, HashMap<u64, SameAs>) {
    let mut fat_fields = HashMap::new();
    let mut copies = HashMap::new();
    let mut fat_cache: HashMap<(String, String), bool> = HashMap::new();
    for block in &graph.blocks {
        for op in &block.operations {
            let Some(result) = op.result.as_ref() else {
                continue;
            };
            match &op.kind {
                OpKind::FieldRead {
                    base, field, pure, ..
                } if field.vec_part.is_none() => {
                    let Some(owner) = field.owner_root.clone() else {
                        continue;
                    };
                    let name = field.name.clone();
                    let is_fat = if let Some(known) = fat_cache.get(&(owner.clone(), name.clone()))
                    {
                        *known
                    } else {
                        let known = field_is_fat_dyn(llbc, &owner, &name);
                        fat_cache.insert((owner, name), known);
                        known
                    };
                    if !is_fat {
                        continue;
                    }
                    fat_fields.entry(result.id()).or_insert(FatField {
                        block: block.id,
                        var_id: result.id(),
                        base: base.clone(),
                        field: field.clone(),
                        pure: *pure,
                    });
                }
                OpKind::UnaryOp {
                    op: name, operand, ..
                } if name == "same_as" => {
                    copies.entry(result.id()).or_insert(SameAs {
                        block: block.id,
                        operand: operand.clone(),
                    });
                }
                // `__cast_instance_intrinsic` is identity at jitcode
                // (`cast_pointer` → `same_as`). The vtable-slot base is that
                // cast of the data word, so follow it back to the fat field.
                OpKind::Call { .. } => {
                    let Some(operand) = cast_instance_operand(&op.kind) else {
                        continue;
                    };
                    copies.entry(result.id()).or_insert(SameAs {
                        block: block.id,
                        operand,
                    });
                }
                _ => {}
            }
        }
    }
    (fat_fields, copies)
}

fn retarget_vtable_method_bases(graph: &mut FunctionGraph, llbc: &Llbc) -> usize {
    let (fat_fields, copies) = index_producers(graph, llbc);
    let sites: Vec<(BlockId, u64, Variable)> = graph
        .blocks
        .iter()
        .flat_map(|block| {
            block.operations.iter().filter_map(|op| {
                let result = op.result.as_ref()?;
                let OpKind::FieldRead { base, field, .. } = &op.kind else {
                    return None;
                };
                if !is_vtable_method(field) {
                    return None;
                }
                Some((block.id, result.id(), base.clone()))
            })
        })
        .collect();
    let mut env = MetaEnv {
        fat_fields,
        copies,
        fat_len: HashMap::new(),
        memo: HashMap::new(),
        visiting: HashSet::new(),
    };
    let mut retargeted = 0usize;
    for (block, result_id, base) in sites {
        let Some(meta) = ensure_meta(graph, &mut env, block, &base) else {
            continue;
        };
        let Some(op) = graph.blocks[block.0].operations.iter_mut().find(|op| {
            op.result
                .as_ref()
                .is_some_and(|result| result.id() == result_id)
        }) else {
            continue;
        };
        let OpKind::FieldRead { base, pure, .. } = &mut op.kind else {
            continue;
        };
        *base = meta;
        // The slot of a vtable does not change. The base is the metadata
        // word, an int (`raw` pointer); `getfield_raw_r` is emitted only
        // for a pure read (`jtransform.py` `rewrite_op_getfield`).
        *pure = true;
        retargeted += 1;
    }
    retargeted
}

fn ensure_meta(
    graph: &mut FunctionGraph,
    env: &mut MetaEnv,
    block: BlockId,
    var: &Variable,
) -> Option<Variable> {
    let key = (block.0, var.id());
    if let Some(found) = env.memo.get(&key) {
        return Some(found.clone());
    }
    if !env.visiting.insert(key) {
        return None;
    }
    let result = ensure_meta_inner(graph, env, block, var);
    env.visiting.remove(&key);
    if let Some(found) = result.clone() {
        env.memo.insert(key, found);
    }
    result
}

fn ensure_meta_inner(
    graph: &mut FunctionGraph,
    env: &mut MetaEnv,
    block: BlockId,
    var: &Variable,
) -> Option<Variable> {
    let fat = env.fat_fields.get(&var.id()).map(|fat| FatField {
        block: fat.block,
        var_id: fat.var_id,
        base: fat.base.clone(),
        field: fat.field.clone(),
        pure: fat.pure,
    });
    if let Some(fat) = fat {
        return Some(emit_fat_len(graph, env, &fat));
    }
    let copy = env
        .copies
        .get(&var.id())
        .map(|copy| (copy.block, copy.operand.clone()));
    if let Some((copy_block, operand)) = copy {
        return ensure_meta(graph, env, copy_block, &operand);
    }
    let inputarg_len = graph.blocks.get(block.0)?.inputargs.len();
    let index = graph.blocks[block.0]
        .inputargs
        .iter()
        .position(|arg| arg == var)?;
    let preds = graph.predecessors(block);
    if preds.is_empty() || preds.iter().any(|pred| *pred == block) {
        return None;
    }
    let mut incoming = Vec::new();
    for pred in &preds {
        let exits = graph.blocks.get(pred.0)?.exits.clone();
        let mut saw = false;
        for (link_index, link) in exits.iter().enumerate() {
            if link.target != block {
                continue;
            }
            if link.args.len() != inputarg_len {
                return None;
            }
            let src = link.args.get(index)?.as_variable()?.clone();
            incoming.push((*pred, link_index, src));
            saw = true;
        }
        if !saw {
            return None;
        }
    }
    if incoming.is_empty() {
        return None;
    }
    let mut metas = Vec::with_capacity(incoming.len());
    for (pred, link_index, src) in &incoming {
        let meta = ensure_meta(graph, env, *pred, src)?;
        metas.push((*pred, *link_index, meta));
    }
    let phi = graph.alloc_value_var_with_type(ConcreteType::Signed);
    graph.push_inputarg_var(block, phi.clone());
    for (pred, link_index, meta) in metas {
        graph.blocks[pred.0].exits[link_index]
            .args
            .push(LinkArg::Value(meta));
    }
    Some(phi)
}

fn emit_fat_len(graph: &mut FunctionGraph, env: &mut MetaEnv, fat: &FatField) -> Variable {
    if let Some(existing) = env.fat_len.get(&fat.var_id) {
        return existing.clone();
    }
    let meta = graph.alloc_value_var_with_type(ConcreteType::Signed);
    let field = fat.field.clone().with_vec_part(VecFieldPart::FatLen);
    let op = SpaceOperation {
        result: Some(meta.clone()),
        kind: OpKind::FieldRead {
            base: fat.base.clone(),
            field,
            ty: ValueType::Int,
            pure: fat.pure,
        },
    };
    let block = &mut graph.blocks[fat.block.0];
    let at = block.operations.iter().position(|op| {
        op.result
            .as_ref()
            .is_some_and(|result| result.id() == fat.var_id)
    });
    match at {
        Some(index) => block.operations.insert(index + 1, op),
        None => block.operations.push(op),
    }
    env.fat_len.insert(fat.var_id, meta.clone());
    meta
}
