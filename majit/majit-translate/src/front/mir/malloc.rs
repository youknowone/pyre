//! Malloc / SROA (`rpython/translator/backendopt/malloc.py`).

use super::*;

/// Replace a live named-struct aggregate with its fields.
///
/// Construction is `p = malloc(S); p.f = v` (`rewrite_op_malloc`). The MIR
/// front first emits a `SyntheticTransparentCtor` so `fuse_boxing_alloc`,
/// `lower_struct_ptr_writes`, and `remove_dead_aggregates` still see the
/// construct-on-stack spelling. After those passes, a remaining named
/// struct is the JIT virtual (`OptVirtualize.make_vstruct`): field reads
/// become the stored SSA values, and a by-value copy does not exist in
/// the source language, so each escape — return, residual call argument,
/// store of the whole value, or phi/link copy — materialises one fresh
/// `New` plus `setfield`s. Mutation of one copy is then a store to that
/// object alone.
///
/// An inlined substructure is the same rule: the value is its fields,
/// never a residual constructor call (`rewrite_op_getsubstruct`,
/// `rewrite_op_getinteriorfield`).
///
/// Aggregates that still participate in a boxing cluster stay
/// constructors: the malloc argument, any phi that carries it, and nested
/// named structs stored into those (the header object fusion reads
/// `ob_type` off). They are the cluster's stack value, not the heap
/// object.
///
/// Closure environments and field-less structs also stay constructors.
/// `rewrite_op_malloc` refuses a size descr whose `all_fielddescrs` is
/// empty, and a closure has no registered layout at all.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded"
)]
pub(in crate::front::mir) fn scalar_replace_named_struct_aggregates(
    graph: &mut FunctionGraph,
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
) -> usize {
    let malloc_args = boxing_cluster_ctor_results(graph);
    let mut rewritten = 0usize;
    loop {
        let Some(site) = next_struct_aggregate_ctor(graph, &malloc_args, struct_field_attrs) else {
            break;
        };
        if !scalar_replace_one_struct_aggregate(graph, site, &malloc_args, struct_field_attrs) {
            break;
        }
        rewritten += 1;
    }
    // `GcStruct('Interior', ('ptr', parent), ('index', Signed))`
    // (`lltype.py` `Ptr._interior_ptr_type_with_index`,
    // `rptr.py` `InteriorPtrRepr.rtype_getitem`) is a malloc between
    // rtyping and `LLTypeMallocRemover`. `flatten` turns `ptr` and
    // `index` into variables; `flowin_op` rewrites `getfield` /
    // `setfield`. An escape keeps the `New`.
    remove_nonescaping_interior_mallocs(graph);
    rewritten
}

pub(in crate::front::mir) struct StructAggregateCtorSite {
    pub(in crate::front::mir) block_idx: usize,
    pub(in crate::front::mir) op_idx: usize,
    pub(in crate::front::mir) result: Variable,
    pub(in crate::front::mir) owner: String,
    /// The allocated STRUCT's identity carried by the constructor call
    /// (`CallTarget::allocated_struct_id`), the TYPE `malloc.py
    /// recreate_malloc` keeps for the recreated malloc.
    pub(in crate::front::mir) owner_id: Option<majit_ir::descr::StructId>,
}

#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded"
)]
pub(in crate::front::mir) fn next_struct_aggregate_ctor(
    graph: &FunctionGraph,
    malloc_args: &std::collections::HashSet<Variable>,
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
) -> Option<StructAggregateCtorSite> {
    for (block_idx, block) in graph.blocks.iter().enumerate() {
        for (op_idx, op) in block.operations.iter().enumerate() {
            let Some(result) = op.result.as_ref() else {
                continue;
            };
            if malloc_args.contains(result) {
                continue;
            }
            let OpKind::Call {
                target:
                    CallTarget::SyntheticTransparentCtor {
                        name,
                        is_struct: true,
                        ..
                    },
                args,
                result_ty,
            } = &op.kind
            else {
                continue;
            };
            if !args.is_empty() {
                continue;
            }
            if majit_charon_reader::ullbc::is_closure_leaf(name) {
                continue;
            }
            // The callee writes the cell, so the post-call field read is
            // not the pre-call SSA value. `optimizeopt/virtualize.py`
            // removes the cell once the callee is inlined.
            if is_mut_ref_cell_name(name) {
                continue;
            }
            let ValueType::Ref(Some(owner)) = result_ty else {
                continue;
            };
            if owner.is_empty() {
                continue;
            }
            let layout_empty =
                struct_field_rows(struct_field_attrs, owner).is_some_and(|rows| rows.is_empty());
            let discovered = struct_ctor_discovered_fields(graph, result);
            if discovered.is_empty() {
                continue;
            }
            if layout_empty && struct_ctor_has_whole_value_use(graph, result) {
                continue;
            }
            let OpKind::Call { target, .. } = &op.kind else {
                unreachable!("matched as a Call above");
            };
            let owner_id = target.allocated_struct_id();
            return Some(StructAggregateCtorSite {
                block_idx,
                op_idx,
                result: result.clone(),
                owner: owner.clone(),
                owner_id,
            });
        }
    }
    None
}

pub(in crate::front::mir) fn struct_ctor_discovered_fields(
    graph: &FunctionGraph,
    result: &Variable,
) -> Vec<(
    crate::model::FieldDescriptor,
    crate::model::LinkArg,
    ValueType,
)> {
    let mut fields: Vec<(
        crate::model::FieldDescriptor,
        crate::model::LinkArg,
        ValueType,
    )> = Vec::new();
    for op in graph.blocks.iter().flat_map(|block| &block.operations) {
        let OpKind::FieldWrite {
            base,
            field,
            value,
            ty,
        } = &op.kind
        else {
            continue;
        };
        if base != result {
            continue;
        }
        if let Some(existing) = fields
            .iter_mut()
            .find(|(seen, _, _)| struct_field_slot_eq(seen, field))
        {
            *existing = (field.clone(), value.clone(), ty.clone());
        } else {
            fields.push((field.clone(), value.clone(), ty.clone()));
        }
    }
    fields
}

pub(in crate::front::mir) fn struct_ctor_has_whole_value_use(
    graph: &FunctionGraph,
    result: &Variable,
) -> bool {
    for block in &graph.blocks {
        if block.inputargs.iter().any(|arg| arg == result) {
            return true;
        }
        match &block.exitswitch {
            Some(ExitSwitch::Value(var)) if var == result => return true,
            Some(ExitSwitch::Fused { args, .. }) if args.iter().any(|arg| arg == result) => {
                return true;
            }
            Some(ExitSwitch::LastException | ExitSwitch::Value(_) | ExitSwitch::Fused { .. })
            | None => {}
        }
        for link in &block.exits {
            if link
                .args
                .iter()
                .any(|arg| arg.as_variable() == Some(result))
            {
                return true;
            }
            if link.last_exception.as_ref().and_then(LinkArg::as_variable) == Some(result)
                || link.last_exc_value.as_ref().and_then(LinkArg::as_variable) == Some(result)
            {
                return true;
            }
        }
        for op in &block.operations {
            if struct_ctor_kind_has_whole_value_use(&op.kind, result) {
                return true;
            }
        }
    }
    false
}

pub(in crate::front::mir) fn struct_ctor_kind_has_whole_value_use(
    kind: &OpKind,
    result: &Variable,
) -> bool {
    match kind {
        OpKind::FieldRead { base, .. } if base == result => false,
        OpKind::FieldWrite { value, .. } => value.as_variable() == Some(result),
        OpKind::Call { target, args, .. } => {
            args.iter().any(|arg| arg.as_variable() == Some(result))
                && !call_target_is_gc_malloc(target)
        }
        OpKind::IndirectCall { args, .. } => args.iter().any(|arg| arg == result),
        kind => crate::inline::op_variable_refs(kind)
            .iter()
            .any(|var| var == result),
    }
}

/// `malloc.py` `LifeTime` — one union-find class of `(block, var)` pairs.
pub(in crate::front::mir) struct MallocLifeTime {
    pub(in crate::front::mir) parent: usize,
    pub(in crate::front::mir) variables: Vec<(usize, Variable)>,
    /// `("op", block, op)` or a non-op tag (`"inputargs"`, `"constant"`,
    /// `"last_exception"`, `"last_exc_value"`).
    pub(in crate::front::mir) creations: Vec<MallocCreation>,
    /// `("op", block, op, index)` or a non-op tag (`"return"`, `"except"`,
    /// `"exitswitch"`, `"dup"`).
    pub(in crate::front::mir) uses: Vec<MallocUse>,
}

pub(in crate::front::mir) enum MallocCreation {
    Op { block: usize, op_idx: usize },
    Other,
}

pub(in crate::front::mir) enum MallocUse {
    Op {
        block: usize,
        op_idx: usize,
        index: usize,
    },
    Other,
}

/// Current value of one flattened field inside `flowin`.
#[derive(Clone)]
pub(in crate::front::mir) enum FieldCur {
    /// `malloc.py` `newvarsmap` entry: a variable or a constant.
    Arg(LinkArg),
    /// Typed zero of a `Ref` / `Str` field (`ConstRefNull`).
    NullRef,
}

/// `malloc.py` `compute_lifetimes` + `_try_inline_malloc` for the lifetime
/// that contains `ctor` (`block_idx`, `result`). `true` when `flowin`
/// removed that malloc. A later `scalar_replace_named_struct_aggregates`
/// iteration is the `remove_simple_mallocs` fixpoint.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded"
)]
pub(in crate::front::mir) fn try_inline_malloc_lifetime(
    graph: &mut FunctionGraph,
    block_idx: usize,
    result: &Variable,
    owner: &str,
    malloc_args: &std::collections::HashSet<Variable>,
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
) -> bool {
    let Some(flatnames) = struct_flatnames(owner, struct_field_attrs) else {
        return false;
    };
    if flatnames.is_empty() {
        return false;
    }
    let lifetimes = compute_lifetimes(graph);
    let Some(info_idx) = lifetimes.index.get(&(block_idx, result.clone())).copied() else {
        return false;
    };
    let info_idx = lifetimes.root_of(info_idx);
    let info = &lifetimes.nodes[info_idx];
    if !lifetime_is_removable(
        graph,
        info,
        owner,
        &flatnames,
        malloc_args,
        struct_field_attrs,
    ) {
        return false;
    }
    let mut by_block: Vec<(usize, Vec<Variable>)> = Vec::new();
    for (block, var) in &info.variables {
        if let Some((_, vars)) = by_block.iter_mut().find(|(seen, _)| seen == block) {
            if !vars.iter().any(|existing| existing == var) {
                vars.push(var.clone());
            }
        } else {
            by_block.push((*block, vec![var.clone()]));
        }
    }
    let zeros: Vec<FieldCur> = flatnames.iter().map(|(_, ty)| zero_field_cur(ty)).collect();
    // `malloc.py` `_try_inline_malloc`: inputargs first, then mallocs
    // created in the block. Each successful malloc increments progress.
    let mut progress = 0usize;
    for (block, vars) in by_block {
        let mut new_map: Option<Vec<FieldCur>> = None;
        let mut new_inputargs: Vec<Variable> = Vec::new();
        let mut inputvars: Vec<Variable> = Vec::new();
        for var in graph.blocks[block].inputargs.clone() {
            if vars.iter().any(|candidate| candidate == &var) {
                inputvars.push(var);
                if new_map.is_none() {
                    let mut map = Vec::with_capacity(flatnames.len());
                    for (_, ty) in &flatnames {
                        let fresh = graph.alloc_value_var_with_type(concretetype_for_field(ty));
                        map.push(FieldCur::Arg(LinkArg::Value(fresh.clone())));
                        new_inputargs.push(fresh);
                    }
                    new_map = Some(map);
                }
            } else {
                new_inputargs.push(var);
            }
        }
        graph.blocks[block].inputargs = new_inputargs;
        if !inputvars.is_empty() {
            flowin_malloc_block(graph, block, inputvars, new_map, &flatnames, &zeros);
        }
        let created: Vec<Variable> = graph.blocks[block]
            .operations
            .iter()
            .filter_map(|op| {
                let produced = op.result.as_ref()?;
                let still_malloc = matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::SyntheticTransparentCtor {
                            is_struct: true,
                            ..
                        },
                        args,
                        ..
                    } if args.is_empty()
                );
                if still_malloc && vars.iter().any(|candidate| candidate == produced) {
                    Some(produced.clone())
                } else {
                    None
                }
            })
            .collect();
        for var in created {
            flowin_malloc_block(graph, block, vec![var], None, &flatnames, &zeros);
            progress += 1;
        }
    }
    // Input-flowin may already have dropped the malloc (`progress` stays 0).
    // The caller treats `false` as "fall back to `New`", so any completed
    // inline reports success. `progress` still counts mallocs removed inside
    // the block, matching `malloc.py` `_try_inline_malloc`.
    let _ = progress;
    true
}

/// `ptr` and `index` of `GcStruct('Interior', …)`
/// (`lltype.py` `Ptr._interior_ptr_type_with_index`).
fn interior_ptr_flatnames() -> Vec<(String, ValueType)> {
    vec![
        ("ptr".to_string(), ValueType::Ref(None)),
        ("index".to_string(), ValueType::Int),
    ]
}

struct InteriorMallocSite {
    block_idx: usize,
    result: Variable,
}

/// Next `New { owner: "Interior" }` whose result is not in `skipped`.
///
/// Not a struct ctor: `next_struct_aggregate_ctor` only sees
/// `SyntheticTransparentCtor`. Folding this `New` into that scan would
/// restore it via `malloc_opkind` when removal fails.
fn next_interior_malloc(graph: &FunctionGraph, skipped: &[Variable]) -> Option<InteriorMallocSite> {
    for (block_idx, block) in graph.blocks.iter().enumerate() {
        for op in &block.operations {
            let OpKind::New { owner } = &op.kind else {
                continue;
            };
            if owner != "Interior" {
                continue;
            }
            let Some(result) = op.result.as_ref() else {
                continue;
            };
            if skipped.iter().any(|seen| seen == result) {
                continue;
            }
            return Some(InteriorMallocSite {
                block_idx,
                result: result.clone(),
            });
        }
    }
    None
}

fn interior_malloc_remains(graph: &FunctionGraph, result: &Variable) -> bool {
    graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(&op.kind, OpKind::New { owner } if owner == "Interior")
                && op.result.as_ref() == Some(result)
        })
    })
}

/// `LLTypeMallocRemover` for `Interior`: every creation is
/// `New { owner: "Interior" }`, and every use is `getfield` / `setfield`
/// of `ptr` or `index` (`flowin_op`). A returned pointer, a store of the
/// word, or a residual argument is `MallocUse::Other` or a non-field op
/// and is not removable. Checked before any rewrite.
fn interior_lifetime_is_removable(
    graph: &FunctionGraph,
    info: &MallocLifeTime,
    flatnames: &[(String, ValueType)],
) -> bool {
    if info.creations.is_empty() {
        return false;
    }
    for creation in &info.creations {
        let MallocCreation::Op { block, op_idx } = creation else {
            return false;
        };
        let Some(op) = graph
            .blocks
            .get(*block)
            .and_then(|block| block.operations.get(*op_idx))
        else {
            return false;
        };
        if !matches!(&op.kind, OpKind::New { owner } if owner == "Interior") {
            return false;
        }
    }
    for use_point in &info.uses {
        let MallocUse::Op {
            block,
            op_idx,
            index,
        } = use_point
        else {
            return false;
        };
        if *index != 0 {
            return false;
        }
        let Some(op) = graph
            .blocks
            .get(*block)
            .and_then(|block| block.operations.get(*op_idx))
        else {
            return false;
        };
        let field_name = match &op.kind {
            OpKind::FieldRead { field, .. } => field.name.as_str(),
            OpKind::FieldWrite {
                field, value, ty, ..
            } => {
                if !field_value_aliasable(value, ty) {
                    return false;
                }
                field.name.as_str()
            }
            _ => return false,
        };
        if !flatnames.iter().any(|(name, _)| name == field_name) {
            return false;
        }
    }
    let mut by_block: Vec<(usize, Vec<Variable>)> = Vec::new();
    for (block, var) in &info.variables {
        if let Some((_, vars)) = by_block.iter_mut().find(|(seen, _)| seen == block) {
            if !vars.iter().any(|existing| existing == var) {
                vars.push(var.clone());
            }
        } else {
            by_block.push((*block, vec![var.clone()]));
        }
    }
    for (block, mut vars) in by_block {
        let mut ready = graph.blocks[block]
            .inputargs
            .iter()
            .any(|var| vars.iter().any(|candidate| candidate == var));
        for op in &graph.blocks[block].operations {
            if let OpKind::UnaryOp {
                op: name, operand, ..
            } = &op.kind
                && name == "same_as"
                && vars.iter().any(|candidate| candidate == operand)
                && let Some(produced) = &op.result
            {
                vars.push(produced.clone());
                continue;
            }
            let refs = crate::inline::op_variable_refs(&op.kind);
            let base_in = refs
                .first()
                .is_some_and(|var| vars.iter().any(|candidate| candidate == var));
            let result_in = op
                .result
                .as_ref()
                .is_some_and(|var| vars.iter().any(|candidate| candidate == var));
            if base_in {
                if !ready {
                    return false;
                }
            } else if result_in {
                ready = true;
            }
        }
    }
    true
}

/// `malloc.py` `_try_inline_malloc` for one `Interior` lifetime.
///
/// `false` leaves the graph untouched. A `true` result may still leave
/// the `New` when `flowin` made no progress; the caller skips that
/// variable instead of selecting it again.
fn try_inline_interior_lifetime(
    graph: &mut FunctionGraph,
    block_idx: usize,
    result: &Variable,
) -> bool {
    let flatnames = interior_ptr_flatnames();
    let lifetimes = compute_lifetimes(graph);
    let Some(info_idx) = lifetimes.index.get(&(block_idx, result.clone())).copied() else {
        return false;
    };
    let info_idx = lifetimes.root_of(info_idx);
    let info = &lifetimes.nodes[info_idx];
    if !interior_lifetime_is_removable(graph, info, &flatnames) {
        return false;
    }
    let mut by_block: Vec<(usize, Vec<Variable>)> = Vec::new();
    for (block, var) in &info.variables {
        if let Some((_, vars)) = by_block.iter_mut().find(|(seen, _)| seen == block) {
            if !vars.iter().any(|existing| existing == var) {
                vars.push(var.clone());
            }
        } else {
            by_block.push((*block, vec![var.clone()]));
        }
    }
    let zeros: Vec<FieldCur> = flatnames.iter().map(|(_, ty)| zero_field_cur(ty)).collect();
    for (block, vars) in by_block {
        let mut new_map: Option<Vec<FieldCur>> = None;
        let mut new_inputargs: Vec<Variable> = Vec::new();
        let mut inputvars: Vec<Variable> = Vec::new();
        for var in graph.blocks[block].inputargs.clone() {
            if vars.iter().any(|candidate| candidate == &var) {
                inputvars.push(var);
                if new_map.is_none() {
                    let mut map = Vec::with_capacity(flatnames.len());
                    for (_, ty) in &flatnames {
                        let fresh = graph.alloc_value_var_with_type(concretetype_for_field(ty));
                        map.push(FieldCur::Arg(LinkArg::Value(fresh.clone())));
                        new_inputargs.push(fresh);
                    }
                    new_map = Some(map);
                }
            } else {
                new_inputargs.push(var);
            }
        }
        graph.blocks[block].inputargs = new_inputargs;
        if !inputvars.is_empty() {
            flowin_malloc_block(graph, block, inputvars, new_map, &flatnames, &zeros);
        }
        let created: Vec<Variable> = graph.blocks[block]
            .operations
            .iter()
            .filter_map(|op| {
                let produced = op.result.as_ref()?;
                let still_malloc = matches!(&op.kind, OpKind::New { owner } if owner == "Interior");
                if still_malloc && vars.iter().any(|candidate| candidate == produced) {
                    Some(produced.clone())
                } else {
                    None
                }
            })
            .collect();
        for var in created {
            flowin_malloc_block(graph, block, vec![var], None, &flatnames, &zeros);
        }
    }
    true
}

/// Remove every non-escaping `Interior` malloc (`LLTypeMallocRemover`).
///
/// A lifetime that fails the check is left as `New` — not rewritten to
/// `malloc_opkind`, and not given the escaping-aggregate copies. A
/// variable whose `New` is still present after the attempt is skipped,
/// so the scan cannot select it again.
fn remove_nonescaping_interior_mallocs(graph: &mut FunctionGraph) {
    let mut skipped: Vec<Variable> = Vec::new();
    loop {
        let Some(site) = next_interior_malloc(graph, &skipped) else {
            break;
        };
        if !try_inline_interior_lifetime(graph, site.block_idx, &site.result)
            || interior_malloc_remains(graph, &site.result)
        {
            skipped.push(site.result);
        }
    }
}

pub(in crate::front::mir) fn struct_flatnames(
    owner: &str,
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
) -> Option<Vec<(String, ValueType)>> {
    struct_field_rows(struct_field_attrs, owner).map(<[_]>::to_vec)
}

/// `malloc.py` `flatten` keys `(S, name)` by the lltype object.
/// Exact `struct_field_attrs` key first, then the one `StructId`
/// (`struct_id_for_name`). Two keys for that id is not one struct.
fn struct_field_rows<'a>(
    struct_field_attrs: &'a std::collections::HashMap<String, Vec<(String, ValueType)>>,
    owner: &str,
) -> Option<&'a [(String, ValueType)]> {
    if let Some(rows) = struct_field_attrs.get(owner) {
        return Some(rows.as_slice());
    }
    let sid = majit_ir::descr::struct_id_for_name(owner)?;
    let mut hit: Option<&[(String, ValueType)]> = None;
    for (key, rows) in struct_field_attrs {
        if key == owner {
            continue;
        }
        let Some(key_sid) = majit_ir::descr::struct_id_for_name(key) else {
            continue;
        };
        if key_sid != sid {
            continue;
        }
        if hit.is_some() {
            return None;
        }
        hit = Some(rows.as_slice());
    }
    hit
}

pub(in crate::front::mir) fn concretetype_for_field(ty: &ValueType) -> crate::model::ConcreteType {
    match ty {
        ValueType::Float => crate::model::ConcreteType::Float,
        ValueType::Void => crate::model::ConcreteType::Void,
        ValueType::Ref(_) | ValueType::Str | ValueType::StringBuilder => {
            crate::model::ConcreteType::GcRef
        }
        ValueType::State | ValueType::Unknown => crate::model::ConcreteType::Unknown,
        ValueType::Int
        | ValueType::Unsigned
        | ValueType::Bool
        | ValueType::SingleFloat
        | ValueType::Int128
        | ValueType::UInt128 => crate::model::ConcreteType::Signed,
    }
}

/// Typed zero, matching `emit_zero_constant_of_ty`. `Ref` is null.
pub(in crate::front::mir) fn zero_field_cur(ty: &ValueType) -> FieldCur {
    match ty {
        ValueType::Ref(_) | ValueType::Str | ValueType::StringBuilder => FieldCur::NullRef,
        ValueType::Bool => FieldCur::Arg(LinkArg::Const(crate::flowspace::model::Constant::new(
            ConstValue::Bool(false),
        ))),
        ValueType::Float => FieldCur::Arg(LinkArg::Const(crate::flowspace::model::Constant::new(
            ConstValue::Float(0),
        ))),
        ValueType::Int128 => FieldCur::Arg(LinkArg::Const(crate::flowspace::model::Constant::new(
            ConstValue::Int128(0),
        ))),
        ValueType::UInt128 => FieldCur::Arg(LinkArg::Const(
            crate::flowspace::model::Constant::new(ConstValue::UInt128(0)),
        )),
        ValueType::Void => FieldCur::Arg(LinkArg::Const(crate::flowspace::model::Constant::new(
            ConstValue::None,
        ))),
        ValueType::SingleFloat => FieldCur::Arg(LinkArg::Const(
            crate::flowspace::model::Constant::new(ConstValue::Int(0)),
        )),
        ValueType::Unsigned => FieldCur::Arg(LinkArg::Const(
            crate::flowspace::model::Constant::new(ConstValue::UInt(0)),
        )),
        ValueType::Int | ValueType::State | ValueType::Unknown => FieldCur::Arg(LinkArg::Const(
            crate::flowspace::model::Constant::new(ConstValue::Int(0)),
        )),
    }
}

pub(in crate::front::mir) fn alias_op_for_field(cur: &FieldCur, ty: &ValueType) -> Option<OpKind> {
    match cur {
        FieldCur::NullRef => Some(OpKind::ConstRefNull),
        FieldCur::Arg(arg) => {
            if matches!(ty, ValueType::Unsigned)
                && matches!(arg, LinkArg::Const(constant) if matches!(constant.value, ConstValue::UInt(0) | ConstValue::Int(0)))
            {
                return Some(OpKind::ConstUInt(0));
            }
            if matches!(ty, ValueType::SingleFloat)
                && matches!(arg, LinkArg::Const(constant) if matches!(constant.value, ConstValue::Int(0)))
            {
                return Some(OpKind::ConstSingleFloat(0));
            }
            link_arg_as_alias_op(arg, ty)
        }
    }
}

pub(in crate::front::mir) fn field_value_aliasable(value: &LinkArg, ty: &ValueType) -> bool {
    alias_op_for_field(&FieldCur::Arg(value.clone()), ty).is_some()
}

/// `malloc.py` `LLTypeMallocRemover.check_malloc` for a front aggregate ctor,
/// the same selection as [`next_struct_aggregate_ctor`].
pub(in crate::front::mir) fn aggregate_ctor_owner(
    graph: &FunctionGraph,
    op: &SpaceOperation,
    malloc_args: &std::collections::HashSet<Variable>,
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
) -> Option<String> {
    let result = op.result.as_ref()?;
    if malloc_args.contains(result) {
        return None;
    }
    let OpKind::Call {
        target:
            CallTarget::SyntheticTransparentCtor {
                name,
                is_struct: true,
                ..
            },
        args,
        result_ty,
    } = &op.kind
    else {
        return None;
    };
    if !args.is_empty() {
        return None;
    }
    if majit_charon_reader::ullbc::is_closure_leaf(name) {
        return None;
    }
    // The callee writes the cell, so the post-call field read is not the
    // pre-call SSA value. `optimizeopt/virtualize.py` removes the cell
    // once the callee is inlined.
    if is_mut_ref_cell_name(name) {
        return None;
    }
    let ValueType::Ref(Some(owner)) = result_ty else {
        return None;
    };
    if owner.is_empty() {
        return None;
    }
    let layout_empty =
        struct_field_rows(struct_field_attrs, owner).is_some_and(|rows| rows.is_empty());
    let discovered = struct_ctor_discovered_fields(graph, result);
    if discovered.is_empty() {
        return None;
    }
    if layout_empty && struct_ctor_has_whole_value_use(graph, result) {
        return None;
    }
    Some(owner.clone())
}

pub(in crate::front::mir) struct MallocLifetimes {
    pub(in crate::front::mir) index: std::collections::HashMap<(usize, Variable), usize>,
    pub(in crate::front::mir) nodes: Vec<MallocLifeTime>,
}

impl MallocLifetimes {
    pub(in crate::front::mir) fn ensure(&mut self, block: usize, var: &Variable) -> usize {
        let key = (block, var.clone());
        if let Some(&idx) = self.index.get(&key) {
            return idx;
        }
        let idx = self.nodes.len();
        self.nodes.push(MallocLifeTime {
            parent: idx,
            variables: vec![key.clone()],
            creations: Vec::new(),
            uses: Vec::new(),
        });
        self.index.insert(key, idx);
        idx
    }

    pub(in crate::front::mir) fn root_of(&self, mut idx: usize) -> usize {
        while self.nodes[idx].parent != idx {
            idx = self.nodes[idx].parent;
        }
        idx
    }

    pub(in crate::front::mir) fn find(&mut self, block: usize, var: &Variable) -> usize {
        let idx = self.ensure(block, var);
        let root = self.root_of(idx);
        let mut cursor = idx;
        while cursor != root {
            let parent = self.nodes[cursor].parent;
            self.nodes[cursor].parent = root;
            cursor = parent;
        }
        root
    }

    pub(in crate::front::mir) fn set_creation(
        &mut self,
        block: usize,
        var: &Variable,
        creation: MallocCreation,
    ) {
        let root = self.find(block, var);
        self.nodes[root].creations.push(creation);
    }

    pub(in crate::front::mir) fn set_use(
        &mut self,
        block: usize,
        var: &Variable,
        use_point: MallocUse,
    ) {
        let root = self.find(block, var);
        self.nodes[root].uses.push(use_point);
    }

    /// `malloc.py` `compute_lifetimes` `union`.
    pub(in crate::front::mir) fn union(
        &mut self,
        block1: usize,
        var1: &Variable,
        block2: usize,
        var2: &Variable,
    ) {
        let left = self.find(block1, var1);
        let right = self.find(block2, var2);
        if left == right {
            return;
        }
        let variables = std::mem::take(&mut self.nodes[right].variables);
        let creations = std::mem::take(&mut self.nodes[right].creations);
        let uses = std::mem::take(&mut self.nodes[right].uses);
        self.nodes[left].variables.extend(variables);
        self.nodes[left].creations.extend(creations);
        self.nodes[left].uses.extend(uses);
        self.nodes[right].parent = left;
    }
}

/// `malloc.py` `BaseMallocRemover.compute_lifetimes`.
pub(in crate::front::mir) fn compute_lifetimes(graph: &FunctionGraph) -> MallocLifetimes {
    let id_to_idx: std::collections::HashMap<BlockId, usize> = graph
        .blocks
        .iter()
        .enumerate()
        .map(|(idx, block)| (block.id, idx))
        .collect();
    let mut lifetimes = MallocLifetimes {
        index: std::collections::HashMap::new(),
        nodes: Vec::new(),
    };
    let start = id_to_idx[&graph.startblock];
    for var in &graph.blocks[start].inputargs {
        lifetimes.set_creation(start, var, MallocCreation::Other);
    }
    let return_block = id_to_idx[&graph.returnblock];
    if let Some(var) = graph.blocks[return_block].inputargs.first() {
        lifetimes.set_use(return_block, var, MallocUse::Other);
    }
    let except_block = id_to_idx[&graph.exceptblock];
    for var in graph.blocks[except_block].inputargs.iter().take(2) {
        lifetimes.set_use(except_block, var, MallocUse::Other);
    }

    let reachable = graph.iterblocks_order();
    for block_id in &reachable {
        let Some(&block_idx) = id_to_idx.get(block_id) else {
            continue;
        };
        let block = &graph.blocks[block_idx];
        for (op_idx, op) in block.operations.iter().enumerate() {
            // `IDENTITY_OPS`: `link_arg_as_alias_op`'s plain `same_as`.
            if let OpKind::UnaryOp {
                op: name, operand, ..
            } = &op.kind
                && name == "same_as"
                && let Some(produced) = &op.result
            {
                lifetimes.union(block_idx, operand, block_idx, produced);
                continue;
            }
            for (index, var) in crate::inline::op_variable_refs(&op.kind).iter().enumerate() {
                lifetimes.set_use(
                    block_idx,
                    var,
                    MallocUse::Op {
                        block: block_idx,
                        op_idx,
                        index,
                    },
                );
            }
            if let Some(produced) = &op.result {
                lifetimes.set_creation(
                    block_idx,
                    produced,
                    MallocCreation::Op {
                        block: block_idx,
                        op_idx,
                    },
                );
            }
        }
        match &block.exitswitch {
            Some(ExitSwitch::Value(var)) => {
                lifetimes.set_use(block_idx, var, MallocUse::Other);
            }
            Some(ExitSwitch::Fused { args, .. }) => {
                for var in args {
                    lifetimes.set_use(block_idx, var, MallocUse::Other);
                }
            }
            Some(ExitSwitch::LastException) | None => {}
        }
    }

    for block_id in &reachable {
        let Some(&block_idx) = id_to_idx.get(block_id) else {
            continue;
        };
        for link in &graph.blocks[block_idx].exits {
            if let Some(var) = link.last_exception.as_ref().and_then(LinkArg::as_variable) {
                lifetimes.set_creation(block_idx, var, MallocCreation::Other);
            }
            if let Some(var) = link.last_exc_value.as_ref().and_then(LinkArg::as_variable) {
                lifetimes.set_creation(block_idx, var, MallocCreation::Other);
            }
            let Some(&target_idx) = id_to_idx.get(&link.target) else {
                continue;
            };
            let target_args = graph.blocks[target_idx].inputargs.clone();
            let mut duplicate_roots: Vec<usize> = Vec::new();
            for (index, arg) in link.args.iter().enumerate() {
                let Some(target_var) = target_args.get(index) else {
                    continue;
                };
                match arg {
                    LinkArg::Value(var) => {
                        lifetimes.union(block_idx, var, target_idx, target_var);
                    }
                    LinkArg::Const(_) => {
                        lifetimes.set_creation(target_idx, target_var, MallocCreation::Other);
                    }
                }
                let Some(var) = arg.as_variable() else {
                    continue;
                };
                let root = lifetimes.find(block_idx, var);
                if duplicate_roots.contains(&root) && lifetimes.nodes[root].creations.len() > 1 {
                    lifetimes.set_use(block_idx, var, MallocUse::Other);
                } else {
                    duplicate_roots.push(root);
                }
            }
        }
    }
    lifetimes
}

pub(in crate::front::mir) fn lifetime_is_removable(
    graph: &FunctionGraph,
    info: &MallocLifeTime,
    owner: &str,
    flatnames: &[(String, ValueType)],
    malloc_args: &std::collections::HashSet<Variable>,
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
) -> bool {
    // `_try_inline_malloc`: every creation point is a malloc of one owner.
    if info.creations.is_empty() {
        return false;
    }
    for creation in &info.creations {
        let MallocCreation::Op { block, op_idx } = creation else {
            return false;
        };
        let Some(op) = graph
            .blocks
            .get(*block)
            .and_then(|block| block.operations.get(*op_idx))
        else {
            return false;
        };
        if aggregate_ctor_owner(graph, op, malloc_args, struct_field_attrs).as_deref()
            != Some(owner)
        {
            return false;
        }
    }
    for use_point in &info.uses {
        let MallocUse::Op {
            block,
            op_idx,
            index,
        } = use_point
        else {
            return false;
        };
        if *index != 0 {
            return false;
        }
        let Some(op) = graph
            .blocks
            .get(*block)
            .and_then(|block| block.operations.get(*op_idx))
        else {
            return false;
        };
        let field_name = match &op.kind {
            OpKind::FieldRead { field, .. } => field.name.as_str(),
            OpKind::FieldWrite {
                field, value, ty, ..
            } => {
                if !field_value_aliasable(value, ty) {
                    return false;
                }
                field.name.as_str()
            }
            _ => return false,
        };
        if !flatnames.iter().any(|(name, _)| name == field_name) {
            return false;
        }
    }
    // A field op before the malloc (and with no incoming field vars) would
    // miss `newvarsmap`. That is not removable (`handle_unreachable` is out
    // of scope).
    let mut by_block: Vec<(usize, Vec<Variable>)> = Vec::new();
    for (block, var) in &info.variables {
        if let Some((_, vars)) = by_block.iter_mut().find(|(seen, _)| seen == block) {
            if !vars.iter().any(|existing| existing == var) {
                vars.push(var.clone());
            }
        } else {
            by_block.push((*block, vec![var.clone()]));
        }
    }
    for (block, mut vars) in by_block {
        let mut ready = graph.blocks[block]
            .inputargs
            .iter()
            .any(|var| vars.iter().any(|candidate| candidate == var));
        for op in &graph.blocks[block].operations {
            if let OpKind::UnaryOp {
                op: name, operand, ..
            } = &op.kind
                && name == "same_as"
                && vars.iter().any(|candidate| candidate == operand)
                && let Some(produced) = &op.result
            {
                vars.push(produced.clone());
                continue;
            }
            let refs = crate::inline::op_variable_refs(&op.kind);
            let base_in = refs
                .first()
                .is_some_and(|var| vars.iter().any(|candidate| candidate == var));
            let result_in = op
                .result
                .as_ref()
                .is_some_and(|var| vars.iter().any(|candidate| candidate == var));
            if base_in {
                if !ready {
                    return false;
                }
            } else if result_in {
                ready = true;
            }
        }
    }
    true
}

/// `malloc.py` `BaseMallocRemover.flowin` + `LLTypeMallocRemover.flowin_op`
/// (`getfield` / `setfield` / `same_as` only).
pub(in crate::front::mir) fn flowin_malloc_block(
    graph: &mut FunctionGraph,
    block: usize,
    mut vars: Vec<Variable>,
    mut newvarsmap: Option<Vec<FieldCur>>,
    flatnames: &[(String, ValueType)],
    zeros: &[FieldCur],
) {
    let ops = std::mem::take(&mut graph.blocks[block].operations);
    let mut newops: Vec<SpaceOperation> = Vec::new();
    for op in ops {
        if let OpKind::UnaryOp {
            op: name, operand, ..
        } = &op.kind
            && name == "same_as"
            && vars.iter().any(|candidate| candidate == operand)
        {
            // `flowin_op` same_as: one flattened list for both pointers.
            if let Some(produced) = &op.result {
                vars.push(produced.clone());
            }
            continue;
        }
        let refs = crate::inline::op_variable_refs(&op.kind);
        let base_in = refs
            .first()
            .is_some_and(|var| vars.iter().any(|candidate| candidate == var));
        let result_in = op
            .result
            .as_ref()
            .is_some_and(|var| vars.iter().any(|candidate| candidate == var));
        if base_in {
            match &op.kind {
                OpKind::FieldRead { field, ty, .. } => {
                    let Some(map) = newvarsmap.as_ref() else {
                        newops.push(op);
                        continue;
                    };
                    let Some(index) = flatnames.iter().position(|(name, _)| name == &field.name)
                    else {
                        newops.push(op);
                        continue;
                    };
                    let Some(kind) = alias_op_for_field(&map[index], ty) else {
                        newops.push(op);
                        continue;
                    };
                    newops.push(SpaceOperation {
                        result: op.result.clone(),
                        kind,
                    });
                }
                OpKind::FieldWrite { field, value, .. } => {
                    let Some(map) = newvarsmap.as_mut() else {
                        newops.push(op);
                        continue;
                    };
                    let Some(index) = flatnames.iter().position(|(name, _)| name == &field.name)
                    else {
                        newops.push(op);
                        continue;
                    };
                    map[index] = FieldCur::Arg(value.clone());
                }
                _ => newops.push(op),
            }
        } else if result_in {
            // Drop the malloc. Field vars start as the typed zeros.
            newvarsmap = Some(zeros.to_vec());
        } else {
            newops.push(op);
        }
    }
    let exits = std::mem::take(&mut graph.blocks[block].exits);
    let mut rewritten = Vec::with_capacity(exits.len());
    for mut link in exits {
        let mut appended = false;
        let mut new_args: Vec<LinkArg> = Vec::new();
        for arg in &link.args {
            if arg
                .as_variable()
                .is_some_and(|var| vars.iter().any(|candidate| candidate == var))
            {
                if !appended {
                    if let Some(map) = &newvarsmap {
                        for (index, cur) in map.iter().enumerate() {
                            new_args.push(field_cur_as_link_arg(
                                graph,
                                &mut newops,
                                cur,
                                &flatnames[index].1,
                            ));
                        }
                    }
                    appended = true;
                }
            } else {
                new_args.push(arg.clone());
            }
        }
        link.args = new_args;
        rewritten.push(link);
    }
    graph.blocks[block].operations = newops;
    graph.blocks[block].exits = rewritten;
}

pub(in crate::front::mir) fn field_cur_as_link_arg(
    graph: &mut FunctionGraph,
    newops: &mut Vec<SpaceOperation>,
    cur: &FieldCur,
    _ty: &ValueType,
) -> LinkArg {
    match cur {
        FieldCur::NullRef => {
            let var = graph.alloc_value_var_with_type(crate::model::ConcreteType::GcRef);
            newops.push(SpaceOperation {
                result: Some(var.clone()),
                kind: OpKind::ConstRefNull,
            });
            LinkArg::Value(var)
        }
        FieldCur::Arg(arg) => arg.clone(),
    }
}

pub(in crate::front::mir) fn scalar_replace_one_struct_aggregate(
    graph: &mut FunctionGraph,
    site: StructAggregateCtorSite,
    malloc_args: &std::collections::HashSet<Variable>,
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
) -> bool {
    let StructAggregateCtorSite {
        block_idx,
        op_idx,
        result,
        owner,
        owner_id,
    } = site;
    let foreign_field_ops = graph.blocks.iter().enumerate().any(|(idx, block)| {
        idx != block_idx
            && block.operations.iter().any(|op| match &op.kind {
                OpKind::FieldRead { base, .. } | OpKind::FieldWrite { base, .. } => base == &result,
                _ => false,
            })
    });
    if foreign_field_ops {
        // `malloc.py` `_try_inline_malloc` / `flowin`: a field op in another
        // block is not an escape when the value only moves along links.
        // Fall back to one `New` per copy when the lifetime is not removable.
        if try_inline_malloc_lifetime(
            graph,
            block_idx,
            &result,
            &owner,
            malloc_args,
            struct_field_attrs,
        ) {
            return true;
        }
        let has_vtable =
            crate::codewriter::heaptracker::attrs_have_vtable(&owner, struct_field_attrs);
        graph.blocks[block_idx].operations[op_idx].kind =
            crate::model::malloc_opkind(owner.clone(), has_vtable, owner_id);
        materialize_whole_value_copies_in_block(
            graph, block_idx, op_idx, &result, &owner, has_vtable,
        );
        return true;
    }

    let has_vtable = crate::codewriter::heaptracker::attrs_have_vtable(&owner, struct_field_attrs);
    // Rolled back when a field read of `result` cannot be retargeted.
    // Dropping the malloc and leaving that read is a use with no def
    // (`rtyper.py` `setconcretetype` / `bindingrepr`).
    let original_ops = graph.blocks[block_idx].operations.clone();
    let ops = std::mem::take(&mut graph.blocks[block_idx].operations);
    let mut out: Vec<crate::model::SpaceOperation> = Vec::new();
    let mut fields: Vec<(
        crate::model::FieldDescriptor,
        crate::model::LinkArg,
        ValueType,
    )> = Vec::new();
    let mut field_alias: Vec<(Variable, Variable)> = Vec::new();
    for (i, op) in ops.into_iter().enumerate() {
        if i == op_idx {
            continue;
        }
        match &op.kind {
            OpKind::FieldWrite {
                base,
                field,
                value,
                ty,
            } if base == &result => {
                upsert_struct_field(&mut fields, field.clone(), value.clone(), ty.clone());
                continue;
            }
            OpKind::FieldRead {
                base, field, ty, ..
            } if base == &result => {
                if let Some((_, value, _)) = fields
                    .iter()
                    .find(|(seen, _, _)| struct_field_slot_eq(seen, field))
                {
                    if let (Some(dest), LinkArg::Value(src)) = (&op.result, value) {
                        // The read is the stored SSA value. A block input
                        // stays a `same_as`: that parameter is distinct.
                        if !graph.blocks[block_idx]
                            .inputargs
                            .iter()
                            .any(|arg| arg == dest)
                        {
                            field_alias.push((dest.clone(), src.clone()));
                            continue;
                        }
                    }
                    if let Some(kind) = link_arg_as_alias_op(value, ty) {
                        out.push(crate::model::SpaceOperation {
                            result: op.result.clone(),
                            kind,
                        });
                        continue;
                    }
                } else if let Some(kind) = project_inlined_substruct_field_read(
                    graph,
                    block_idx,
                    &out,
                    &owner,
                    &fields,
                    &op,
                    struct_field_attrs,
                    &original_ops,
                    &result,
                ) {
                    out.push(crate::model::SpaceOperation {
                        result: op.result.clone(),
                        kind,
                    });
                    continue;
                }
            }
            _ => {}
        }
        if struct_ctor_kind_has_whole_value_use(&op.kind, &result) {
            let kind = replace_whole_value_uses_in_kind(
                graph, &mut out, &op.kind, &result, &owner, &fields, has_vtable,
            );
            out.push(crate::model::SpaceOperation {
                result: op.result.clone(),
                kind,
            });
        } else {
            out.push(op);
        }
    }

    let ctor_still_used = out.iter().any(|op| {
        crate::inline::op_variable_refs(&op.kind)
            .iter()
            .any(|var| var == &result)
    });
    if ctor_still_used {
        // The leaf is still a read of this aggregate. Keep the malloc
        // (`rewrite_op_malloc`) so `setconcretetype` has a defining op.
        graph.blocks[block_idx].operations = original_ops;
        graph.blocks[block_idx].operations[op_idx].kind =
            crate::model::malloc_opkind(owner.clone(), has_vtable);
        materialize_whole_value_copies_in_block(
            graph, block_idx, op_idx, &result, &owner, has_vtable,
        );
        return true;
    }

    let apply_alias = |var: &Variable| -> Variable {
        let mut current = var.clone();
        let mut seen: Vec<Variable> = Vec::new();
        loop {
            if seen.iter().any(|item| item == &current) {
                return current;
            }
            seen.push(current.clone());
            let Some((_, next)) = field_alias.iter().find(|(from, _)| from == &current) else {
                return current;
            };
            current = next.clone();
        }
    };
    for op in &mut out {
        let remap = |var: &Variable| apply_alias(var);
        op.kind = crate::inline::remap_op_kind(&op.kind, &remap);
        if let Some(result) = op.result.clone() {
            let renamed = apply_alias(&result);
            if renamed != result {
                op.result = Some(renamed);
            }
        }
    }
    let mut phi_copies: Vec<(BlockId, usize, Variable)> = Vec::new();
    let mut exits = std::mem::take(&mut graph.blocks[block_idx].exits);
    for link in &mut exits {
        for arg in &mut link.args {
            if let Some(var) = arg.as_variable() {
                let renamed = apply_alias(var);
                if &renamed != var {
                    *arg = LinkArg::Value(renamed);
                }
            }
        }
        if let Some(var) = link.last_exception.as_ref().and_then(LinkArg::as_variable) {
            let renamed = apply_alias(var);
            if &renamed != var {
                link.last_exception = Some(LinkArg::Value(renamed));
            }
        }
        if let Some(var) = link.last_exc_value.as_ref().and_then(LinkArg::as_variable) {
            let renamed = apply_alias(var);
            if &renamed != var {
                link.last_exc_value = Some(LinkArg::Value(renamed));
            }
        }
    }
    for link in &mut exits {
        for (slot, arg) in link.args.iter_mut().enumerate() {
            if arg.as_variable() != Some(&result) {
                continue;
            }
            let copy = emit_materialized_struct_copy(graph, &mut out, &owner, &fields, has_vtable);
            *arg = LinkArg::Value(copy.clone());
            // Each predecessor passes its own copy. The successor parameter
            // stays one variable; renaming it to a single predecessor's copy
            // and leaving the body on the old name made every later edge
            // read the first copy.
            phi_copies.push((link.target, slot, copy));
        }
        if link.last_exception.as_ref().and_then(LinkArg::as_variable) == Some(&result) {
            let copy = emit_materialized_struct_copy(graph, &mut out, &owner, &fields, has_vtable);
            link.last_exception = Some(LinkArg::Value(copy));
        }
        if link.last_exc_value.as_ref().and_then(LinkArg::as_variable) == Some(&result) {
            let copy = emit_materialized_struct_copy(graph, &mut out, &owner, &fields, has_vtable);
            link.last_exc_value = Some(LinkArg::Value(copy));
        }
    }
    graph.blocks[block_idx].operations = out;
    graph.blocks[block_idx].exits = exits;
    bind_struct_phi_copies(graph, &result, &phi_copies);
    true
}

/// When the successor parameter *is* the aggregate variable, point that
/// parameter and every use in the successor at one phi. Predecessors keep
/// the distinct copies already stored in their link args.
pub(in crate::front::mir) fn bind_struct_phi_copies(
    graph: &mut FunctionGraph,
    from: &Variable,
    copies: &[(BlockId, usize, Variable)],
) {
    let mut groups: Vec<(BlockId, usize, Vec<Variable>)> = Vec::new();
    for (target, slot, copy) in copies {
        if let Some((_, _, incoming)) = groups
            .iter_mut()
            .find(|(seen_target, seen_slot, _)| seen_target == target && *seen_slot == *slot)
        {
            incoming.push(copy.clone());
        } else {
            groups.push((*target, *slot, vec![copy.clone()]));
        }
    }
    for (target, slot, incoming) in groups {
        let Some(idx) = graph.blocks.iter().position(|block| block.id == target) else {
            continue;
        };
        if graph.blocks[idx].inputargs.get(slot) != Some(from) {
            continue;
        }
        let phi = if incoming.len() == 1 {
            incoming[0].clone()
        } else {
            graph.alloc_value_var()
        };
        let block = &mut graph.blocks[idx];
        block.inputargs[slot] = phi.clone();
        remap_struct_phi_uses(block, from, &phi);
    }
}

pub(in crate::front::mir) fn remap_struct_phi_uses(
    block: &mut Block,
    from: &Variable,
    to: &Variable,
) {
    let remap = |var: &Variable| {
        if var == from { to.clone() } else { var.clone() }
    };
    for op in &mut block.operations {
        op.kind = crate::inline::remap_op_kind(&op.kind, &remap);
    }
    match &mut block.exitswitch {
        Some(ExitSwitch::Value(var)) if var == from => *var = to.clone(),
        Some(ExitSwitch::Fused { args, .. }) => {
            for arg in args {
                if arg == from {
                    *arg = to.clone();
                }
            }
        }
        Some(ExitSwitch::LastException | ExitSwitch::Value(_)) | None => {}
    }
    for link in &mut block.exits {
        for arg in &mut link.args {
            retarget_link_arg(arg, from, to);
        }
        if let Some(arg) = link.last_exception.as_mut() {
            retarget_link_arg(arg, from, to);
        }
        if let Some(arg) = link.last_exc_value.as_mut() {
            retarget_link_arg(arg, from, to);
        }
    }
}

pub(in crate::front::mir) fn retarget_link_arg(arg: &mut LinkArg, from: &Variable, to: &Variable) {
    if arg.as_variable() == Some(from) {
        *arg = LinkArg::Value(to.clone());
    }
}

/// Replace uses of `from` with `to` and drop the op that defined `from`.
///
/// This is the graph `remove_same_as` builds. Returns false when `from`
/// is a block input: that parameter is the edge's own variable, and the
/// copy that feeds it has to stay a distinct `same_as`.
pub(crate) fn forward_identity(graph: &mut FunctionGraph, from: &Variable, to: &Variable) -> bool {
    if from == to {
        return true;
    }
    if graph
        .blocks
        .iter()
        .any(|block| block.inputargs.iter().any(|arg| arg == from))
    {
        return false;
    }
    let remap = |var: &Variable| {
        if var == from { to.clone() } else { var.clone() }
    };
    for block in &mut graph.blocks {
        block
            .operations
            .retain(|op| op.result.as_ref() != Some(from));
        for op in &mut block.operations {
            op.kind = crate::inline::remap_op_kind(&op.kind, &remap);
        }
        match &mut block.exitswitch {
            Some(ExitSwitch::Value(var)) if var == from => *var = to.clone(),
            Some(ExitSwitch::Fused { args, .. }) => {
                for arg in args {
                    if arg == from {
                        *arg = to.clone();
                    }
                }
            }
            Some(ExitSwitch::LastException | ExitSwitch::Value(_)) | None => {}
        }
        for link in &mut block.exits {
            for arg in &mut link.args {
                retarget_link_arg(arg, from, to);
            }
            if let Some(arg) = link.last_exception.as_mut() {
                retarget_link_arg(arg, from, to);
            }
            if let Some(arg) = link.last_exc_value.as_mut() {
                retarget_link_arg(arg, from, to);
            }
        }
    }
    true
}

pub(in crate::front::mir) fn upsert_struct_field(
    fields: &mut Vec<(
        crate::model::FieldDescriptor,
        crate::model::LinkArg,
        ValueType,
    )>,
    field: crate::model::FieldDescriptor,
    value: crate::model::LinkArg,
    ty: ValueType,
) {
    if let Some(existing) = fields
        .iter_mut()
        .find(|(seen, _, _)| struct_field_slot_eq(seen, &field))
    {
        *existing = (field, value, ty);
    } else {
        fields.push((field, value, ty));
    }
}

/// Last-write-wins is per inlined slot. `FatData` and `FatMeta` of one
/// attribute are two words of `two_word_fat_struct` (`rclass.py`
/// `OBJECT.typeptr`); they must not collapse to one store.
fn struct_field_slot_eq(
    a: &crate::model::FieldDescriptor,
    b: &crate::model::FieldDescriptor,
) -> bool {
    a.name == b.name && a.vec_part == b.vec_part
}

/// Read of an inlined by-value substructure (`malloc.py`
/// `LLTypeMallocRemover` `getsubstruct` arm, `jtransform.py`
/// `rewrite_op_getsubstruct`).
///
/// A dotted `prefix.rest` is the leaf of the value stored in `prefix`.
/// An offset-0 hop keeps the inner descr (`nest_inline_leaf`,
/// `Transformer.zero_contents`); that descr's owner is the nested
/// struct, so the read moves onto the one stored value of that struct.
fn project_inlined_substruct_field_read(
    graph: &FunctionGraph,
    block_idx: usize,
    emitted: &[SpaceOperation],
    aggregate_owner: &str,
    fields: &[(FieldDescriptor, LinkArg, ValueType)],
    op: &SpaceOperation,
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
    original_ops: &[SpaceOperation],
    aggregate: &Variable,
) -> Option<OpKind> {
    let OpKind::FieldRead {
        field, ty, pure, ..
    } = &op.kind
    else {
        return None;
    };
    if let Some((prefix, rest)) = field.name.split_once('.')
        && !prefix.is_empty()
        && !rest.is_empty()
    {
        let (_, value, stored_ty) = fields
            .iter()
            .find(|(seen, _, _)| seen.name == prefix && seen.vec_part.is_none())?;
        let LinkArg::Value(src) = value else {
            return None;
        };
        // `flowin_op` reads `op.args[0].concretetype.TO`. The stored
        // field type is that STRUCT when the write recorded `Ref(S)`.
        // A phi of a residual call can carry the struct with no `New`
        // (`newvarstype` never ran). The dotted leaves are `S._flds`
        // (`flatten`). One struct with that field list is S. When two
        // structs share the list, the container's instantiated type
        // argument is `FIELDTYPE` (`substitute_spill_typevars`).
        let owner = if let ValueType::Ref(Some(stored)) = stored_ty
            && !stored.is_empty()
        {
            stored.clone()
        } else {
            match struct_value_owner(graph, block_idx, emitted, src) {
                Some(owner) => owner,
                None => {
                    let leaves = dotted_prefix_leaves(original_ops, aggregate, prefix)?;
                    unique_nested_struct_owner(struct_field_attrs, &leaves, aggregate_owner)?
                }
            }
        };
        let mut projected = field.clone();
        projected.name = rest.to_string();
        projected.owner_root = Some(owner);
        // The container's StructId is not the nested value's.
        projected.owner_id = None;
        projected.owner_declared_gc = None;
        return Some(OpKind::FieldRead {
            base: src.clone(),
            field: projected,
            ty: ty.clone(),
            pure: *pure,
        });
    }
    let nested = field.owner_root.as_deref()?;
    if struct_ids_equal(nested, aggregate_owner) {
        return None;
    }
    let mut found: Option<&Variable> = None;
    for (_, value, _) in fields {
        let LinkArg::Value(src) = value else {
            continue;
        };
        let Some(owner) = struct_value_owner(graph, block_idx, emitted, src) else {
            continue;
        };
        if !struct_ids_equal(&owner, nested) {
            continue;
        }
        if found.is_some() {
            return None;
        }
        found = Some(src);
    }
    let src = match found {
        Some(src) => src.clone(),
        // Offset 0 keeps the inner descr (`nest_inline_leaf`,
        // `Transformer.zero_contents`) and drops the container field
        // name. The one stored value no same-named read consumed is
        // that substructure (`flatten`: a ContainerType that is not a
        // primitive flatname).
        None => sole_unconsumed_stored_value(fields, original_ops, aggregate)
            .filter(|_| {
                struct_flatnames(nested, struct_field_attrs).is_some_and(|rows| {
                    rows.iter()
                        .any(|(name, fty)| name == &field.name && fty == ty)
                })
            })?
            .clone(),
    };
    Some(OpKind::FieldRead {
        base: src,
        field: field.clone(),
        ty: ty.clone(),
        pure: *pure,
    })
}

/// Dotted `prefix.rest` reads of `aggregate`, as `(rest, ty)`.
///
/// A deeper dot is not one field of a flat nested struct (`S._flds`
/// keys are single names). Disagreeing types for one name are not one
/// field either.
fn dotted_prefix_leaves(
    ops: &[SpaceOperation],
    aggregate: &Variable,
    prefix: &str,
) -> Option<Vec<(String, ValueType)>> {
    let mut leaves: Vec<(String, ValueType)> = Vec::new();
    for op in ops {
        let OpKind::FieldRead {
            base, field, ty, ..
        } = &op.kind
        else {
            continue;
        };
        if base != aggregate {
            continue;
        }
        let Some((seen_prefix, rest)) = field.name.split_once('.') else {
            continue;
        };
        if seen_prefix != prefix || rest.is_empty() {
            continue;
        }
        if rest.contains('.') {
            return None;
        }
        if let Some((_, prev)) = leaves.iter().find(|(name, _)| name == rest) {
            if prev != ty {
                return None;
            }
            continue;
        }
        leaves.push((rest.to_string(), ty.clone()));
    }
    if leaves.is_empty() {
        None
    } else {
        Some(leaves)
    }
}

/// The one struct whose fields are exactly `leaves`.
///
/// `malloc.py` `flatten` takes `FIELDTYPE = S._flds[name]` off the
/// value's concretetype and keys `(S, name)` by the lltype object
/// (`StructId`). Several structs, or a struct with extra fields, is
/// not that type. Spellings that share one `StructId` are one hit;
/// the representative is the first sorted spelling.
///
/// Two different structs can share one field list. The container's
/// instantiated spelling still names the substituted type variable
/// (`collect_by_value_struct_leaves` / `substitute_spill_typevars`):
/// that argument is `FIELDTYPE`. A hit has to be one of those
/// arguments. Zero or several remaining hits is not a type.
fn unique_nested_struct_owner(
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
    leaves: &[(String, ValueType)],
    container: &str,
) -> Option<String> {
    if leaves.is_empty() {
        return None;
    }
    let mut matched: Vec<&str> = Vec::new();
    for (owner, fields) in struct_field_attrs {
        if fields.len() != leaves.len() {
            continue;
        }
        let same = leaves
            .iter()
            .all(|(name, ty)| fields.iter().any(|(fname, fty)| fname == name && fty == ty))
            && fields
                .iter()
                .all(|(fname, fty)| leaves.iter().any(|(name, ty)| fname == name && fty == ty));
        if same {
            matched.push(owner.as_str());
        }
    }
    matched.sort_unstable();
    let mut groups: Vec<String> = Vec::new();
    for owner in matched {
        if groups.iter().any(|prev| struct_ids_equal(prev, owner)) {
            continue;
        }
        groups.push(owner.to_string());
    }
    if groups.len() == 1 {
        return groups.pop();
    }
    let args = top_level_generic_args(container);
    if args.is_empty() {
        return None;
    }
    groups.retain(|owner| args.iter().any(|arg| struct_ids_equal(owner, arg)));
    if groups.len() == 1 {
        groups.pop()
    } else {
        None
    }
}

/// Top-level type arguments of an instantiated spelling
/// (`Outer<Inner, *mut T>`). Nested `<>` stay inside one argument.
/// No argument list is empty.
fn top_level_generic_args(owner: &str) -> Vec<&str> {
    let Some(open) = owner.find('<') else {
        return Vec::new();
    };
    let mut depth = 1usize;
    let mut start = open + 1;
    let mut args = Vec::new();
    for (i, ch) in owner[open + 1..].char_indices() {
        let idx = open + 1 + i;
        match ch {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    let arg = owner[start..idx].trim();
                    if !arg.is_empty() {
                        args.push(arg);
                    }
                    break;
                }
            }
            ',' if depth == 1 => {
                let arg = owner[start..idx].trim();
                if !arg.is_empty() {
                    args.push(arg);
                }
                start = idx + 1;
            }
            _ => {}
        }
    }
    args
}

/// Stored value no same-named field read of `aggregate` consumed.
///
/// `flowin_op` `getfield` of a primitive uses `newvarsmap[S, name]`
/// directly. A by-value substructure keeps its own variable
/// (`needsubmallocs` / the `getsubstruct` arm). Two such values, or a
/// constant, are not one substructure.
fn sole_unconsumed_stored_value<'a>(
    fields: &'a [(FieldDescriptor, LinkArg, ValueType)],
    ops: &[SpaceOperation],
    aggregate: &Variable,
) -> Option<&'a Variable> {
    let mut found: Option<&Variable> = None;
    for (field, value, _) in fields {
        let consumed = ops.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { base, field: read, .. }
                    if base == aggregate && struct_field_slot_eq(field, read)
            )
        });
        if consumed {
            continue;
        }
        let LinkArg::Value(src) = value else {
            return None;
        };
        if found.is_some() {
            return None;
        }
        found = Some(src);
    }
    found
}

/// How `var` is defined, for an inlined-substructure retarget.
enum StructValueDef {
    /// `malloc` of a named struct (`rewrite_op_malloc` / `New`).
    Struct(String),
    /// `same_as` of another value.
    Alias(Variable),
    /// Defined, and not a struct value.
    Other,
}

/// `New` / struct-ctor owner of `var`.
///
/// A block input is the phi of its incoming link args (`malloc.py`
/// `compute_lifetimes` flows one flattened value across the edge).
/// Every incoming struct owner must agree. A `same_as` is not a
/// second object (`remove_same_as`).
fn struct_value_owner(
    graph: &FunctionGraph,
    block_idx: usize,
    emitted: &[SpaceOperation],
    var: &Variable,
) -> Option<String> {
    struct_value_owner_rec(graph, block_idx, emitted, var, &mut Vec::new())
}

fn struct_value_owner_rec(
    graph: &FunctionGraph,
    block_idx: usize,
    emitted: &[SpaceOperation],
    var: &Variable,
    seen: &mut Vec<Variable>,
) -> Option<String> {
    if seen.iter().any(|item| item == var) {
        return None;
    }
    seen.push(var.clone());
    if let Some(def) = struct_value_def(emitted, var) {
        return struct_def_owner(graph, block_idx, emitted, def, seen);
    }
    for (idx, block) in graph.blocks.iter().enumerate() {
        if idx == block_idx {
            continue;
        }
        if let Some(def) = struct_value_def(&block.operations, var) {
            return struct_def_owner(graph, block_idx, emitted, def, seen);
        }
    }
    phi_struct_owner(graph, block_idx, emitted, var, seen)
}

fn struct_def_owner(
    graph: &FunctionGraph,
    block_idx: usize,
    emitted: &[SpaceOperation],
    def: StructValueDef,
    seen: &mut Vec<Variable>,
) -> Option<String> {
    match def {
        StructValueDef::Struct(owner) => Some(owner),
        StructValueDef::Alias(next) => {
            struct_value_owner_rec(graph, block_idx, emitted, &next, seen)
        }
        StructValueDef::Other => None,
    }
}

fn phi_struct_owner(
    graph: &FunctionGraph,
    block_idx: usize,
    emitted: &[SpaceOperation],
    var: &Variable,
    seen: &mut Vec<Variable>,
) -> Option<String> {
    let mut incoming: Vec<Variable> = Vec::new();
    for block in &graph.blocks {
        let Some(slot) = block.inputargs.iter().position(|arg| arg == var) else {
            continue;
        };
        let target = block.id;
        for pred in &graph.blocks {
            for link in &pred.exits {
                if link.target != target {
                    continue;
                }
                match link.args.get(slot) {
                    Some(LinkArg::Value(src)) => incoming.push(src.clone()),
                    _ => return None,
                }
            }
        }
    }
    let mut agreed: Option<String> = None;
    for src in &incoming {
        if seen.iter().any(|item| item == src) {
            continue;
        }
        let owner = struct_value_owner_rec(graph, block_idx, emitted, src, seen)?;
        match &agreed {
            None => agreed = Some(owner),
            Some(prev) if struct_ids_equal(prev, &owner) => {}
            Some(_) => return None,
        }
    }
    agreed
}

fn struct_value_def(ops: &[SpaceOperation], var: &Variable) -> Option<StructValueDef> {
    for op in ops {
        if op.result.as_ref() != Some(var) {
            continue;
        }
        return Some(match &op.kind {
            OpKind::New { owner } | OpKind::NewWithVtable { owner, .. } => {
                StructValueDef::Struct(owner.clone())
            }
            OpKind::Call {
                target:
                    CallTarget::SyntheticTransparentCtor {
                        is_struct: true, ..
                    },
                result_ty: ValueType::Ref(Some(owner)),
                ..
            } => StructValueDef::Struct(owner.clone()),
            OpKind::UnaryOp { op, operand, .. } if op == "same_as" => {
                StructValueDef::Alias(operand.clone())
            }
            _ => StructValueDef::Other,
        });
    }
    None
}

/// `malloc.py` `flatten` keys `(S, name)` by the lltype object.
/// Unregistered spellings are equal only to themselves: two `None`
/// lookups are not one struct.
fn struct_ids_equal(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    match (
        majit_ir::descr::struct_id_for_name(a),
        majit_ir::descr::struct_id_for_name(b),
    ) {
        (Some(left), Some(right)) => left == right,
        _ => false,
    }
}

pub(in crate::front::mir) fn link_arg_as_alias_op(
    value: &crate::model::LinkArg,
    result_ty: &ValueType,
) -> Option<OpKind> {
    match value {
        LinkArg::Value(var) => Some(OpKind::UnaryOp {
            op: "same_as".to_string(),
            operand: var.clone(),
            result_ty: result_ty.clone(),
        }),
        LinkArg::Const(constant) => match &constant.value {
            crate::flowspace::model::ConstValue::Int(v) => Some(OpKind::ConstInt(*v)),
            crate::flowspace::model::ConstValue::UInt(v) => Some(OpKind::ConstUInt(*v)),
            crate::flowspace::model::ConstValue::Int128(v) => Some(OpKind::ConstInt128(*v)),
            crate::flowspace::model::ConstValue::UInt128(v) => Some(OpKind::ConstUInt128(*v)),
            crate::flowspace::model::ConstValue::Float(v) => Some(OpKind::ConstFloat(*v)),
            crate::flowspace::model::ConstValue::Bool(v) => Some(OpKind::ConstBool(*v)),
            crate::flowspace::model::ConstValue::None => Some(OpKind::ConstNone),
            crate::flowspace::model::ConstValue::ByteStr(v) => Some(OpKind::ConstStr(v.clone())),
            _ => None,
        },
    }
}

fn frame_box_payload_arg<'a>(
    fields: &'a [(
        crate::model::FieldDescriptor,
        crate::model::LinkArg,
        ValueType,
    )],
) -> Option<&'a crate::model::LinkArg> {
    fields
        .iter()
        .find(|(field, _, _)| field.name == "owner_root")
        .or_else(|| fields.iter().find(|(field, _, _)| field.name == "ptr"))
        .map(|(_, value, _)| value)
}

fn error_carrier_payload_arg<'a>(
    fields: &'a [(
        crate::model::FieldDescriptor,
        crate::model::LinkArg,
        ValueType,
    )],
) -> Option<&'a crate::model::LinkArg> {
    fields
        .iter()
        .find(|(field, _, _)| field.name == "__pos_0" || field.name == "0")
        .or_else(|| fields.first())
        .map(|(_, value, _)| value)
}

pub(in crate::front::mir) fn emit_materialized_struct_copy(
    graph: &mut FunctionGraph,
    out: &mut Vec<crate::model::SpaceOperation>,
    owner: &str,
    fields: &[(
        crate::model::FieldDescriptor,
        crate::model::LinkArg,
        ValueType,
    )],
    has_vtable: bool,
) -> Variable {
    // `createframe` returns the GC `PyFrame`. A by-value escape of the
    // host handle is that GCREF, not a second malloc of the wrapper.
    // `OperationError` is the exception instance; the host `PyError`
    // handle is that same GCREF.
    let payload = if frame_box::owner_is_frame_box(owner) {
        frame_box_payload_arg(fields)
    } else if owner_is_error_carrier(owner) {
        error_carrier_payload_arg(fields)
    } else {
        None
    };
    if let Some(payload) = payload {
        match payload {
            LinkArg::Value(var) => return var.clone(),
            other => {
                if let Some(kind) = link_arg_as_alias_op(other, &ValueType::Ref(None)) {
                    let result = graph.alloc_value_var();
                    out.push(crate::model::SpaceOperation {
                        result: Some(result.clone()),
                        kind,
                    });
                    return result;
                }
            }
        }
    }
    let result = graph.alloc_value_var();
    // A materialized copy has no constructor call; every field row of the
    // copied value names the same STRUCT (`heaptracker.all_fielddescrs`).
    let mut owner_ids = fields.iter().filter_map(|(field, _, _)| {
        (field.owner_root.as_deref() == Some(owner))
            .then_some(field.owner_id)
            .flatten()
    });
    let owner_id = owner_ids.next();
    debug_assert!(
        owner_ids.all(|other| Some(other) == owner_id),
        "materialized copy of {owner} mixes field STRUCT ids"
    );
    out.push(crate::model::SpaceOperation {
        result: Some(result.clone()),
        kind: crate::model::malloc_opkind(owner.to_string(), has_vtable, owner_id),
    });
    for (field, value, ty) in fields {
        // Keep the typeptr store on the copy. `fuse_boxing_alloc`
        // (`resolve_header_plan`) reads the dominating constant `ob_type`
        // write to fill `NewWithVtable.vtable`; `jtransform.py
        // rewrite_op_setfield` / `is_typeptr_getset` drops it after that
        // capture. Dropping it here leaves `malloc_typed` residual.
        out.push(crate::model::SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: result.clone(),
                field: field.clone(),
                value: value.clone(),
                ty: ty.clone(),
            },
        });
    }
    result
}

pub(in crate::front::mir) fn replace_whole_value_uses_in_kind(
    graph: &mut FunctionGraph,
    out: &mut Vec<crate::model::SpaceOperation>,
    kind: &OpKind,
    result: &Variable,
    owner: &str,
    fields: &[(
        crate::model::FieldDescriptor,
        crate::model::LinkArg,
        ValueType,
    )],
    has_vtable: bool,
) -> OpKind {
    match kind {
        OpKind::Call {
            target,
            args,
            result_ty,
        } => {
            // `FunctionReprBase.call` builds one `vlist` and passes those
            // variables to the single `direct_call` / `indirect_call`.
            let copy = emit_materialized_struct_copy(graph, out, owner, fields, has_vtable);
            let args = args
                .iter()
                .map(|arg| {
                    if arg.as_variable() == Some(result) {
                        LinkArg::Value(copy.clone())
                    } else {
                        arg.clone()
                    }
                })
                .collect();
            OpKind::Call {
                target: target.clone(),
                args,
                result_ty: result_ty.clone(),
            }
        }
        OpKind::IndirectCall {
            funcptr,
            args,
            graphs,
            family_key,
            result_ty,
        } => {
            // `get_eval_fn` is an `indirect_call` (`rpbc.py FunctionReprBase.call`).
            // A by-value struct argument is a stack address in the walker C
            // frame unless materialised. One `New` gives the residual a heap
            // pointer, the same escape `Call` already takes.
            // `FunctionReprBase.call` builds one `vlist` and passes those
            // variables to the single `indirect_call`; a repeated argument is
            // the same object.
            let copy = emit_materialized_struct_copy(graph, out, owner, fields, has_vtable);
            let args = args
                .iter()
                .map(|arg| {
                    if arg == result {
                        copy.clone()
                    } else {
                        arg.clone()
                    }
                })
                .collect();
            OpKind::IndirectCall {
                funcptr: funcptr.clone(),
                args,
                graphs: graphs.clone(),
                family_key: family_key.clone(),
                result_ty: result_ty.clone(),
            }
        }
        OpKind::FieldWrite {
            base,
            field,
            value,
            ty,
        } if value.as_variable() == Some(result) => OpKind::FieldWrite {
            base: base.clone(),
            field: field.clone(),
            value: LinkArg::Value(emit_materialized_struct_copy(
                graph, out, owner, fields, has_vtable,
            )),
            ty: ty.clone(),
        },
        other => {
            let copy = emit_materialized_struct_copy(graph, out, owner, fields, has_vtable);
            crate::inline::remap_op_kind(other, &|var| {
                if var == result {
                    copy.clone()
                } else {
                    var.clone()
                }
            })
        }
    }
}

pub(in crate::front::mir) fn materialize_whole_value_copies_in_block(
    graph: &mut FunctionGraph,
    block_idx: usize,
    ctor_idx: usize,
    result: &Variable,
    owner: &str,
    has_vtable: bool,
) {
    let ops = std::mem::take(&mut graph.blocks[block_idx].operations);
    let mut out: Vec<crate::model::SpaceOperation> = Vec::new();
    let mut fields: Vec<(
        crate::model::FieldDescriptor,
        crate::model::LinkArg,
        ValueType,
    )> = Vec::new();
    for (i, op) in ops.into_iter().enumerate() {
        if let OpKind::FieldWrite {
            base,
            field,
            value,
            ty,
        } = &op.kind
            && base == result
        {
            upsert_struct_field(&mut fields, field.clone(), value.clone(), ty.clone());
        }
        if i > ctor_idx && struct_ctor_kind_has_whole_value_use(&op.kind, result) {
            let kind = replace_whole_value_uses_in_kind(
                graph, &mut out, &op.kind, result, owner, &fields, has_vtable,
            );
            out.push(crate::model::SpaceOperation {
                result: op.result.clone(),
                kind,
            });
        } else {
            out.push(op);
        }
    }
    let mut exits = std::mem::take(&mut graph.blocks[block_idx].exits);
    for link in &mut exits {
        for arg in &mut link.args {
            if arg.as_variable() != Some(result) {
                continue;
            }
            let copy = emit_materialized_struct_copy(graph, &mut out, owner, &fields, has_vtable);
            *arg = LinkArg::Value(copy);
        }
    }
    graph.blocks[block_idx].operations = out;
    graph.blocks[block_idx].exits = exits;
}

/// Variables that still belong to an unfused boxing cluster: each
/// `lltype::malloc[_typed*]` argument, the phis that carry it, and every
/// nested named-struct stored into those aggregates.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded"
)]
pub(in crate::front::mir) fn boxing_cluster_ctor_results(
    graph: &FunctionGraph,
) -> std::collections::HashSet<Variable> {
    use std::collections::HashSet;

    let mut cluster: HashSet<Variable> = HashSet::new();
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::Call { target, args, .. } = &op.kind else {
                continue;
            };
            if !call_target_is_gc_malloc(target) {
                continue;
            }
            cluster.extend(args.iter().filter_map(LinkArg::as_variable).cloned());
        }
    }
    let mut incoming: std::collections::HashMap<crate::model::BlockId, Vec<(usize, usize)>> =
        std::collections::HashMap::new();
    for (pred_idx, pred) in graph.blocks.iter().enumerate() {
        for (link_idx, link) in pred.exits.iter().enumerate() {
            incoming
                .entry(link.target)
                .or_default()
                .push((pred_idx, link_idx));
        }
    }
    let mut growing = true;
    while growing {
        growing = false;
        for block in &graph.blocks {
            for (slot, arg) in block.inputargs.iter().enumerate() {
                if !cluster.contains(arg) {
                    continue;
                }
                let Some(preds) = incoming.get(&block.id) else {
                    continue;
                };
                for &(pred_idx, link_idx) in preds {
                    let link = &graph.blocks[pred_idx].exits[link_idx];
                    if let Some(v) = link.args.get(slot).and_then(LinkArg::as_variable)
                        && cluster.insert(v.clone())
                    {
                        growing = true;
                    }
                }
            }
            for op in &block.operations {
                let OpKind::FieldWrite { base, value, .. } = &op.kind else {
                    continue;
                };
                if !cluster.contains(base) {
                    continue;
                }
                if let Some(v) = value.as_variable()
                    && cluster.insert(v.clone())
                {
                    growing = true;
                }
            }
        }
    }
    cluster
}
