//! Iterator adapters → the loop they denote.
//!
//! ## Positioning
//!
//! `Enumerate::next` / `StepBy::next` / `Wtf8CodePoints::next` are foreign
//! (`opacity: Foreign`, body `Opaque` in the extracted ULLBC) — the same
//! class as `core::slice::iter::<Impl>::next`, which `front::iter_next`
//! already rewrites at the *call site* into the native `next` op.  RPython
//! has no iterator adapters (`rlist.py` / `rrange.py` lower explicit loops),
//! so an adapter is the loop it denotes, not a residual callee and not a
//! new opcode.
//!
//! `Enumerate { iter, count }` is
//! ```text
//!     let a = self.iter.next()?;
//!     let i = self.count;
//!     self.count += 1;
//!     Some((i, a))
//! ```
//! (`core::iter::adapters::enumerate::Enumerate::next`).  This pass
//! replaces the Opaque `enumerate(it)` constructor with a two-field pair
//! `{iter, count: 0}` the annotator can project, retargets `Enumerate::next`
//! at the inner list iterator, and packs `(count, item)` on the Some arm
//! after `front::iter_next` folds the inner `next` into StopIteration.
//!
//! Fail-safe: a site whose inner iterator is not a list `iter` op, or
//! whose Option match is not the for-loop diamond, is left as the residual
//! call (census Skip).
//!
//! `Map::collect` is the same class of foreign adapter, but the predicate
//! lives on the **construction site**, not inside the polymorphic shim:
//! the caller holds a concrete closure ADT, so this pass rewrites
//! `map(it, f).collect()` into the loop it denotes and calls that
//! closure's inherent `call_mut` on a reborrow of the env.  `FilterMap::next` / `Iter::position`
//! still stay residual — they need an early-exit match on the predicate
//! result, which is a different CFG.

use crate::flowspace::model::{ConstValue, Variable};
use crate::front::bool_then::{close_goto_mixed, reproduce_exit_args};
use crate::front::iter_next::{
    BackEdges, iter_op_container_with, originates_from_iter_op, walk_back_to_source, walk_back_with,
};
use crate::front::option_map_or::emit_narrow;
use crate::front::result_exc::back_substitute;
use crate::model::{
    BlockId, CallTarget, ExitCase, ExitSwitch, FieldDescriptor, FunctionGraph, Link, LinkArg,
    OpKind, SpaceOperation, ValueType,
};

/// Owner leaf of the pair that stands in for an `Enumerate` adapter.
/// Distinct from a source `Tuple` so the count slot cannot union with a
/// yield-tuple `__pos_1`.
pub(crate) const ENUMERATE_PAIR_OWNER: &str = "__majit_enumerate";

/// Recognised `Enumerate::next` residual — FunctionPath of the foreign
/// adapter impl, or a Method whose receiver leaf is `Enumerate`.
pub(crate) fn is_enumerate_next_target(target: &CallTarget) -> bool {
    match target {
        CallTarget::Method {
            name,
            receiver_root,
            ..
        } => name == "next" && receiver_root.as_deref() == Some("Enumerate"),
        CallTarget::FunctionPath { segments, .. } => {
            adapter_path_ends_with(segments, "enumerate", "next")
        }
        _ => false,
    }
}

/// `Iterator::enumerate(it)` / `it.enumerate()` — the constructor whose
/// result is the adapter the `next` site iterates.
pub(crate) fn is_enumerate_ctor_target(target: &CallTarget) -> bool {
    match target {
        CallTarget::Method { name, .. } => name == "enumerate",
        CallTarget::FunctionPath { segments, .. } => {
            segments.last().map(String::as_str) == Some("enumerate")
        }
        CallTarget::SyntheticTransparentCtor {
            name, owner_path, ..
        } => name == "Enumerate" || owner_path.last().map(String::as_str) == Some("Enumerate"),
        _ => false,
    }
}

fn adapter_path_ends_with(segments: &[String], adapter: &str, leaf: &str) -> bool {
    segments.last().map(String::as_str) == Some(leaf)
        && segments
            .windows(2)
            .any(|w| w[0] == "adapters" && w[1] == adapter)
}

/// The inner iterator an `Enumerate` value wraps, when the backward walk
/// reaches `enumerate(it)` or an `Enumerate { iter, count }` aggregate.
pub(crate) fn inner_iter_of_enumerate(
    graph: &FunctionGraph,
    edges: &BackEdges,
    enum_var: &Variable,
) -> Option<Variable> {
    if let Some(inner) = walk_back_with(graph, edges, enum_var, |op| match &op.kind {
        OpKind::Call { target, args, .. }
            if is_enumerate_ctor_target(target) && args.len() == 1 =>
        {
            Some(args[0].clone().into_variable())
        }
        _ => None,
    }) {
        return Some(inner);
    }
    let origin = walk_back_with(graph, edges, enum_var, |op| match &op.kind {
        OpKind::Call {
            target: CallTarget::SyntheticTransparentCtor { .. },
            ..
        } => op.result.clone(),
        _ => None,
    })?;
    field_write_value(graph, &origin, &["iter", "__pos_0"])
}

fn field_write_value(graph: &FunctionGraph, base: &Variable, names: &[&str]) -> Option<Variable> {
    for op in graph.blocks.iter().flat_map(|b| &b.operations) {
        let OpKind::FieldWrite {
            base: write_base,
            field,
            value: LinkArg::Value(value),
            ..
        } = &op.kind
        else {
            continue;
        };
        if write_base == base && names.contains(&field.name.as_str()) {
            return Some(value.clone());
        }
    }
    None
}

fn pair_ctor_target() -> CallTarget {
    CallTarget::synthetic_transparent_struct_ctor(
        vec![ENUMERATE_PAIR_OWNER.to_string()],
        ENUMERATE_PAIR_OWNER,
    )
}

fn pair_field(name: &str) -> FieldDescriptor {
    FieldDescriptor::new(name, Some(ENUMERATE_PAIR_OWNER.to_string()))
}

fn emit_pair_field_write(
    base: &Variable,
    name: &str,
    value: LinkArg,
    ty: ValueType,
) -> SpaceOperation {
    SpaceOperation {
        result: None,
        kind: OpKind::FieldWrite {
            base: base.clone(),
            field: pair_field(name),
            value,
            ty,
        },
    }
}

/// Replace `e = enumerate(it)` (or an `Enumerate` aggregate) with
/// `{iter: it, count: 0}` under [`ENUMERATE_PAIR_OWNER`].  The pair is
/// mutated in place each iteration, so the count does not need a new
/// loop-carried SSA slot.
pub(crate) fn rewrite_enumerate_ctor_to_pair(
    graph: &mut FunctionGraph,
    edges: &BackEdges,
    enum_var: &Variable,
    inner: &Variable,
) -> Result<(), String> {
    let name = graph.name.clone();
    // `enum_var` is often the loop-header phi; the constructor lives on
    // the entry edge.  Rewrite that origin op, not the phi.  `edges`
    // indexes the graph as it stands on entry; the walk runs before this
    // function's first mutation.
    let origin = walk_back_with(graph, edges, enum_var, |op| match &op.kind {
        OpKind::Call { target, .. } if is_enumerate_ctor_target(target) => op.result.clone(),
        _ => None,
    })
    .ok_or_else(|| format!("{name}: enumerate value has no constructor origin"))?;
    let (bi, oi) = graph
        .blocks
        .iter()
        .enumerate()
        .find_map(|(bi, b)| {
            b.operations
                .iter()
                .position(|op| op.result.as_ref() == Some(&origin))
                .map(|oi| (bi, oi))
        })
        .ok_or_else(|| format!("{name}: enumerate constructor origin has no producer op"))?;
    let enum_var = &origin;
    let is_call_ctor = matches!(
        &graph.blocks[bi].operations[oi].kind,
        OpKind::Call { target, args, .. }
            if is_enumerate_ctor_target(target) && args.len() == 1
    );
    let is_aggregate = matches!(
        &graph.blocks[bi].operations[oi].kind,
        OpKind::Call {
            target: CallTarget::SyntheticTransparentCtor { name, .. },
            ..
        } if name == "Enumerate" || name == ENUMERATE_PAIR_OWNER
    );
    if !is_call_ctor && !is_aggregate {
        return Err(format!(
            "{name}: enumerate origin is not enumerate(it) or an Enumerate aggregate"
        ));
    }
    if is_call_ctor {
        let inner_arg = match &graph.blocks[bi].operations[oi].kind {
            OpKind::Call { args, .. } => args[0].clone(),
            _ => unreachable!("is_call_ctor"),
        };
        graph.blocks[bi].operations[oi] = SpaceOperation {
            result: Some(enum_var.clone()),
            kind: OpKind::Call {
                target: pair_ctor_target(),
                args: Vec::new(),
                result_ty: ValueType::Ref(Some(ENUMERATE_PAIR_OWNER.into())),
            },
        };
        let zero = graph.alloc_value_var();
        graph.blocks[bi].operations.insert(
            oi + 1,
            SpaceOperation {
                result: Some(zero.clone()),
                kind: OpKind::ConstInt(0),
            },
        );
        graph.blocks[bi].operations.insert(
            oi + 2,
            emit_pair_field_write(enum_var, "iter", inner_arg, ValueType::Ref(None)),
        );
        graph.blocks[bi].operations.insert(
            oi + 3,
            emit_pair_field_write(enum_var, "count", LinkArg::Value(zero), ValueType::Unsigned),
        );
        return Ok(());
    }
    graph.blocks[bi].operations[oi].kind = OpKind::Call {
        target: pair_ctor_target(),
        args: Vec::new(),
        result_ty: ValueType::Ref(Some(ENUMERATE_PAIR_OWNER.into())),
    };
    for op in &mut graph.blocks[bi].operations {
        if let OpKind::FieldWrite {
            base, field, ty, ..
        } = &mut op.kind
            && base == enum_var
        {
            field.owner_root = Some(ENUMERATE_PAIR_OWNER.to_string());
            if field.name == "iter" {
                *ty = ValueType::Ref(None);
            } else if field.name == "count" {
                *ty = ValueType::Unsigned;
            }
        }
    }
    if field_write_value(graph, enum_var, &["iter"]).is_none() {
        graph.blocks[bi].operations.insert(
            oi + 1,
            emit_pair_field_write(
                enum_var,
                "iter",
                LinkArg::Value(inner.clone()),
                ValueType::Ref(None),
            ),
        );
    }
    if field_write_value(graph, enum_var, &["count"]).is_none() {
        let zero = graph.alloc_value_var();
        graph.blocks[bi].operations.insert(
            oi + 1,
            SpaceOperation {
                result: Some(zero.clone()),
                kind: OpKind::ConstInt(0),
            },
        );
        graph.blocks[bi].operations.insert(
            oi + 2,
            emit_pair_field_write(enum_var, "count", LinkArg::Value(zero), ValueType::Unsigned),
        );
    }
    Ok(())
}

/// Insert `inner = pair.iter` immediately before the residual `next` in
/// block `a`.  Returns the inner iterator variable and the (shifted)
/// index of the `next` op.
pub(crate) fn insert_pair_iter_read(
    graph: &mut FunctionGraph,
    a: usize,
    next_idx: usize,
    pair: &Variable,
) -> (Variable, usize) {
    let inner = graph.alloc_value_var();
    graph.blocks[a].operations.insert(
        next_idx,
        SpaceOperation {
            result: Some(inner.clone()),
            kind: OpKind::FieldRead {
                base: pair.clone(),
                field: pair_field("iter"),
                ty: ValueType::Ref(None),
                pure: false,
            },
        },
    );
    (inner, next_idx + 1)
}

/// `SomeTuple(items)` (`annotator/model.py`) and `TupleRepr`
/// (`rtyper/rtuple.py`, chosen by `rtyper.getrepr`) are one shape per
/// item-repr list. The `(count, item)` tuple uses that shape as its owner.
pub(crate) fn enumerate_yield_owner(item_ty: &ValueType) -> String {
    let atom = match item_ty {
        ValueType::Str => "String",
        ValueType::Int => "isize",
        ValueType::Unsigned => "usize",
        ValueType::Float => "f64",
        ValueType::SingleFloat => "f32",
        ValueType::Bool => "bool",
        ValueType::Int128 => "i128",
        ValueType::UInt128 => "u128",
        ValueType::Ref(Some(root)) => root.as_str(),
        ValueType::Ref(None) => "Ptr",
        ValueType::StringBuilder => "StringBuilder",
        ValueType::Void => "()",
        ValueType::State => "State",
        ValueType::Unknown => "Unknown",
    };
    format!("Tuple<usize,{atom}>")
}

fn paint_pos_owner(graph: &mut FunctionGraph, block: usize, base: &Variable, owner: &str) {
    for op in &mut graph.blocks[block].operations {
        let field = match &mut op.kind {
            OpKind::FieldRead {
                base: read_base,
                field,
                ..
            }
            | OpKind::FieldWrite {
                base: read_base,
                field,
                ..
            } if read_base == base && field.name.starts_with("__pos_") => field,
            _ => continue,
        };
        field.owner_root = Some(owner.to_string());
    }
}

/// On the Some arm: increment `pair.count` and pack `(count, item)` as
/// the value the residual `opt.__pos_0` read named — the adapter's
/// `Some((i, a))`.
pub(crate) fn pack_enumerate_payload(
    graph: &mut FunctionGraph,
    some_target: usize,
    item: &Variable,
    item_ty: &ValueType,
    pair: &Variable,
    name: &str,
) -> Result<Variable, String> {
    let count = graph.alloc_value_var();
    let one = graph.alloc_value_var();
    let new_count = graph.alloc_value_var();
    let tup = graph.alloc_value_var();
    let owner = enumerate_yield_owner(item_ty);
    let element = graph.blocks[some_target].operations.iter().find_map(|op| {
        let OpKind::FieldRead { base, field, ty, .. } = &op.kind else {
            return None;
        };
        (base == item && field.name == "__pos_1")
            .then(|| op.result.clone().map(|result| (result, ty.clone())))
            .flatten()
    });
    let mut prefix = vec![
        SpaceOperation {
            result: Some(count.clone()),
            kind: OpKind::FieldRead {
                base: pair.clone(),
                field: pair_field("count"),
                ty: ValueType::Unsigned,
                pure: false,
            },
        },
        SpaceOperation {
            result: Some(one.clone()),
            kind: OpKind::ConstInt(1),
        },
        SpaceOperation {
            result: Some(new_count.clone()),
            kind: OpKind::BinOp {
                op: "add".to_string(),
                lhs: count.clone(),
                rhs: one,
                result_ty: ValueType::Unsigned,
            },
        },
        emit_pair_field_write(
            pair,
            "count",
            LinkArg::Value(new_count),
            ValueType::Unsigned,
        ),
        SpaceOperation {
            result: Some(tup.clone()),
            kind: OpKind::Call {
                target: CallTarget::synthetic_transparent_ctor(&owner),
                args: Vec::new(),
                result_ty: ValueType::Ref(Some(owner.clone())),
            },
        },
        SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: tup.clone(),
                field: FieldDescriptor::new("__pos_0", Some(owner.clone())),
                value: LinkArg::Value(count),
                ty: ValueType::Unsigned,
            },
        },
    ];
    let pos1_write = |value: Variable, ty: ValueType| SpaceOperation {
        result: None,
        kind: OpKind::FieldWrite {
            base: tup.clone(),
            field: FieldDescriptor::new("__pos_1", Some(owner.clone())),
            value: LinkArg::Value(value),
            ty,
        },
    };
    if element.is_none() {
        prefix.push(pos1_write(item.clone(), item_ty.clone()));
    }
    // Decide before the prefix is spliced in. The `__pos_1` write below
    // names `item`, so a use-count taken afterwards would see that write
    // and refuse the unread `for _ in` arm.
    let has_pos0 = graph.blocks[some_target].operations.iter().any(|op| {
        matches!(
            &op.kind,
            OpKind::FieldRead { base, field, .. }
                if base == item && field.name == "__pos_0"
        )
    });
    prefix.append(&mut graph.blocks[some_target].operations);
    graph.blocks[some_target].operations = prefix;
    if let Some((value, ty)) = element {
        let at = graph.blocks[some_target]
            .operations
            .iter()
            .position(|op| op.result.as_ref() == Some(&value))
            .expect("item read precedes its copy onto the packed tuple");
        graph.blocks[some_target]
            .operations
            .insert(at + 1, pos1_write(value, ty));
    }
    if has_pos0 {
        collapse_pos0_onto(graph, some_target, item, &tup, name)?;
    }
    paint_pos_owner(graph, some_target, &tup, &owner);
    Ok(tup)
}

/// The Some arm forwarded the old payload. Replace that exit value with
/// the packed tuple when the successor's only use of the slot is a
/// `__pos_N` read, and paint those reads with the packed owner.
pub(crate) fn rewrite_forwarded_payload(
    graph: &mut FunctionGraph,
    some_block: usize,
    carrier: &Variable,
    packed: &Variable,
    owner: &str,
) {
    // Validation already declined a merge or an exitswitch on the slot.
    // Re-check so a caller cannot paint one predecessor of a merge.
    if !successor_reads_packed_payload(graph, some_block, carrier) {
        return;
    }
    let forwards: Vec<(usize, usize)> = graph.blocks[some_block]
        .exits
        .iter()
        .enumerate()
        .flat_map(|(exit_i, link)| {
            link.args.iter().enumerate().filter_map(move |(arg_i, arg)| {
                matches!(arg, LinkArg::Value(v) if v == carrier).then_some((exit_i, arg_i))
            })
        })
        .collect();
    for (exit_i, arg_i) in forwards {
        let target = graph.blocks[some_block].exits[exit_i].target.0;
        graph.blocks[some_block].exits[exit_i].args[arg_i] = LinkArg::Value(packed.clone());
        if let Some(slot) = graph.blocks[target].inputargs.get(arg_i).cloned() {
            paint_pos_owner(graph, target, &slot, owner);
        }
    }
}

/// `true` when every forward of `carrier` lands on a successor input whose
/// only uses are `__pos_N` reads. A bare forward, or any other use, keeps
/// the old payload shape reachable and must decline.
pub(crate) fn successor_reads_packed_payload(
    graph: &FunctionGraph,
    some_block: usize,
    carrier: &Variable,
) -> bool {
    let mut saw = false;
    for link in &graph.blocks[some_block].exits {
        for (arg_i, arg) in link.args.iter().enumerate() {
            let LinkArg::Value(value) = arg else {
                continue;
            };
            if value != carrier {
                continue;
            }
            saw = true;
            let target_idx = link.target.0;
            let preds = graph
                .blocks
                .iter()
                .flat_map(|block| &block.exits)
                .filter(|link| link.target.0 == target_idx)
                .count();
            // A merge still has another predecessor passing the old shape.
            if preds != 1 {
                return false;
            }
            let target = &graph.blocks[target_idx];
            let Some(slot) = target.inputargs.get(arg_i) else {
                return false;
            };
            let mut read = false;
            for op in &target.operations {
                let OpKind::FieldRead { base, field, .. } = &op.kind else {
                    if crate::front::result_exc::op_operand_vars(&op.kind).contains(slot) {
                        return false;
                    }
                    continue;
                };
                if base == slot {
                    if !field.name.starts_with("__pos_") {
                        return false;
                    }
                    read = true;
                }
            }
            match &target.exitswitch {
                Some(ExitSwitch::Value(switched)) if switched == slot => return false,
                Some(ExitSwitch::Fused { args, .. }) if args.contains(slot) => return false,
                _ => {}
            }
            if target.exits.iter().any(|link| {
                link.args
                    .iter()
                    .any(|arg| matches!(arg, LinkArg::Value(v) if v == slot))
            }) {
                return false;
            }
            if !read {
                return false;
            }
        }
    }
    saw
}



/// `collapse_pos0_read` but the payload is `onto`, not the Option slot
/// itself: after packing, `opt.__pos_0` is the `(i, a)` tuple.
fn collapse_pos0_onto(
    graph: &mut FunctionGraph,
    some_target: usize,
    carrier: &Variable,
    onto: &Variable,
    name: &str,
) -> Result<(), String> {
    let read_idx = graph.blocks[some_target].operations.iter().position(|op| {
        matches!(
            &op.kind,
            OpKind::FieldRead { base, field, .. }
                if base == carrier && field.name == "__pos_0"
        )
    });
    let Some(read_idx) = read_idx else {
        return Err(format!(
            "{name}: enumerate Some arm has no __pos_0 read; the caller checked one"
        ));
    };
    let read_result = graph.blocks[some_target].operations[read_idx]
        .result
        .clone()
        .ok_or_else(|| format!("{name}: __pos_0 read without result"))?;
    graph.blocks[some_target].operations.remove(read_idx);
    let onto = onto.clone();
    let rename = |v: &Variable| -> Variable {
        if *v == read_result {
            onto.clone()
        } else {
            v.clone()
        }
    };
    let block = &mut graph.blocks[some_target];
    for op in &mut block.operations {
        op.kind = crate::inline::remap_op_kind(&op.kind, &rename);
    }
    let (sw, exits) = crate::model::remap_control_flow_metadata_var(
        &block.exitswitch,
        &block.exits,
        rename,
        |b| b,
    );
    block.exitswitch = sw;
    block.exits = exits;
    Ok(())
}

/// Look up an `Enumerate::next` residual and, if its inner iterator is a
/// list `iter` op, return the adapter value and that inner iterator.
/// Does not mutate; the caller validates the diamond first.
pub(crate) fn enumerate_list_inner(
    graph: &FunctionGraph,
    edges: &BackEdges,
    next_target: &CallTarget,
    enum_var: &Variable,
) -> Result<Option<Variable>, String> {
    if !is_enumerate_next_target(next_target) {
        return Ok(None);
    }
    let inner = inner_iter_of_enumerate(graph, edges, enum_var).ok_or_else(|| {
        format!(
            "{}: Enumerate::next iterator does not originate from enumerate(it)",
            graph.name
        )
    })?;
    if iter_op_container_with(graph, edges, &inner).is_none() {
        return Err(format!(
            "{}: Enumerate inner iterator does not originate from an iter op",
            graph.name
        ));
    }
    Ok(Some(inner))
}

/// A recognized `Map::collect` construction site.  Types are resolved
/// while the receiver `Map<I, F>` and destination `Vec<U>` are still
/// MIR `TyRef`s; the post-pass only needs the closure's inherent
/// `call_once` owner and the inner-item / collected-element kinds.
#[derive(Clone)]
pub(crate) struct MapCollectSite {
    /// The `collect` call result (the `Vec`) — locates block A.
    pub result_var: Variable,
    /// The closure env ADT `name_path` — the `call_once` inherent-method owner.
    pub call_once_owner: String,
    /// `I::Item` projected to a [`ValueType`] — the `next` element and the
    /// `(x,)` args-tuple payload the synthesized `call_once` writes.
    pub payload_ty: ValueType,
    /// Concrete class of a reference payload, matching
    /// `option_closure_select::ClosureSelectSite::payload_class_root`.
    pub payload_class_root: Option<String>,
    /// `<X>` suffix for the closure `Args` tuple `(payload,)`.
    pub args_tuple_suffix: String,
    /// `F::Output` / `Vec<U>` element kind — the `call_once` result and
    /// the `Vec::push` item.
    pub call_result_ty: ValueType,
    /// Element kind recorded beside the synthesized `next` so
    /// `front::iter_next` can pick the list vs range answer.
    pub inner_item_ty: ValueType,
}

/// `true` iff `path` names the `Map` adapter ADT
/// (`core::iter::adapters::map::Map`).
pub(crate) fn is_map_adapter_path(path: &str) -> bool {
    path.ends_with("::iter::adapters::map::Map")
}

/// Residual `Map::collect` / `Iterator::collect` on a `Map` adapter.
/// The capture site still proves the receiver is `Map` and the dest is
/// `Vec`; this only names the leaf so a `HashMap::collect` Method is
/// not rejected before that type gate.
pub(crate) fn is_map_collect_target(target: &CallTarget) -> bool {
    match target {
        CallTarget::Method { name, .. } => name == "collect",
        CallTarget::FunctionPath { segments, .. } => {
            segments.last().map(String::as_str) == Some("collect")
        }
        _ => false,
    }
}

pub(crate) fn is_map_ctor_target(target: &CallTarget) -> bool {
    match target {
        CallTarget::Method { name, .. } => name == "map",
        CallTarget::FunctionPath { segments, .. } => {
            segments.last().map(String::as_str) == Some("map")
        }
        CallTarget::SyntheticTransparentCtor {
            name, owner_path, ..
        } => name == "Map" || owner_path.last().map(String::as_str) == Some("Map"),
        _ => false,
    }
}

fn originates_from_range_ctor(graph: &FunctionGraph, var: &Variable) -> bool {
    walk_back_to_source(graph, var, |op| match &op.kind {
        OpKind::Call {
            target:
                CallTarget::SyntheticTransparentCtor {
                    name, owner_path, ..
                },
            ..
        } if name == "Range" || owner_path.last().map(String::as_str) == Some("Range") => Some(()),
        _ => None,
    })
    .is_some()
}

fn inner_is_list_or_range(graph: &FunctionGraph, inner: &Variable) -> bool {
    originates_from_iter_op(graph, inner) || originates_from_range_ctor(graph, inner)
}

fn unique_predecessor(graph: &FunctionGraph, block: usize) -> Result<usize, String> {
    let mut found = None;
    for (i, b) in graph.blocks.iter().enumerate() {
        for link in &b.exits {
            if link.target.0 == block {
                if found.is_some() {
                    return Err(format!(
                        "{}: collect block {block} has multiple predecessors",
                        graph.name
                    ));
                }
                found = Some(i);
            }
        }
    }
    found.ok_or_else(|| format!("{}: collect block {block} has no predecessor", graph.name))
}

fn producer_in_block<'a>(
    graph: &'a FunctionGraph,
    block: usize,
    var: &Variable,
) -> Option<(usize, &'a SpaceOperation)> {
    graph.blocks[block]
        .operations
        .iter()
        .enumerate()
        .find(|(_, op)| op.result.as_ref() == Some(var))
}

struct MapCtor {
    block: usize,
    op_idx: usize,
    inner: Variable,
    env: Variable,
    mapped: Variable,
    aggregate: bool,
}

fn map_ctor_from_op(op: &SpaceOperation, op_idx: usize, block: usize) -> Option<MapCtor> {
    match &op.kind {
        OpKind::Call { target, args, .. } if is_map_ctor_target(target) && args.len() == 2 => {
            Some(MapCtor {
                block,
                op_idx,
                inner: args[0].clone().into_variable(),
                env: args[1].clone().into_variable(),
                mapped: op.result.clone()?,
                aggregate: false,
            })
        }
        OpKind::Call {
            target:
                CallTarget::SyntheticTransparentCtor {
                    name, owner_path, ..
                },
            ..
        } if name == "Map" || owner_path.last().map(String::as_str) == Some("Map") => {
            Some(MapCtor {
                block,
                op_idx,
                inner: Variable::new(),
                env: Variable::new(),
                mapped: op.result.clone()?,
                aggregate: true,
            })
        }
        _ => None,
    }
}

fn locate_map_ctor(
    graph: &FunctionGraph,
    collect_block: usize,
    mapped: &Variable,
) -> Result<MapCtor, String> {
    let name = graph.name.clone();
    if let Some((idx, op)) = producer_in_block(graph, collect_block, mapped)
        && let Some(mut ctor) = map_ctor_from_op(op, idx, collect_block)
    {
        if ctor.aggregate {
            ctor.inner = field_write_value(graph, &ctor.mapped, &["iter", "__pos_0"])
                .ok_or_else(|| format!("{name}: Map aggregate has no iter field write"))?;
            ctor.env = field_write_value(graph, &ctor.mapped, &["f", "__pos_1"])
                .ok_or_else(|| format!("{name}: Map aggregate has no closure field write"))?;
        }
        return Ok(ctor);
    }
    let m = unique_predecessor(graph, collect_block)?;
    let mapped_m = back_substitute(graph, &[(m, collect_block)], mapped, &name)?;
    let (idx, op) = producer_in_block(graph, m, &mapped_m).ok_or_else(|| {
        format!("{name}: Map adapter has no constructor in the collect predecessor")
    })?;
    let mut ctor = map_ctor_from_op(op, idx, m)
        .ok_or_else(|| format!("{name}: collect argument does not originate from map(it, f)"))?;
    if ctor.aggregate {
        ctor.inner = field_write_value(graph, &ctor.mapped, &["iter", "__pos_0"])
            .ok_or_else(|| format!("{name}: Map aggregate has no iter field write"))?;
        ctor.env = field_write_value(graph, &ctor.mapped, &["f", "__pos_1"])
            .ok_or_else(|| format!("{name}: Map aggregate has no closure field write"))?;
    }
    Ok(ctor)
}

fn var_in_block_scope(graph: &FunctionGraph, block: usize, var: &Variable) -> bool {
    graph.blocks[block].inputargs.iter().any(|v| v == var)
        || graph.blocks[block]
            .operations
            .iter()
            .any(|op| op.result.as_ref() == Some(var))
}

fn vec_new_call(result: Variable) -> SpaceOperation {
    SpaceOperation {
        result: Some(result),
        kind: OpKind::Call {
            target: CallTarget::function_path(["vec", "Vec", "new"]),
            args: Vec::new(),
            result_ty: ValueType::Ref(None),
        },
    }
}

/// `FnMut::call_mut(&mut env, (payload,))`. The receiver is a `same_as`
/// reborrow of the loop-carried env, so the iteration does not consume it.
fn emit_call_mut(
    graph: &mut FunctionGraph,
    block: BlockId,
    env: Variable,
    arg: Option<(Variable, ValueType, Option<String>)>,
    call_owner: &str,
    result_ty: ValueType,
    args_tuple_suffix: &str,
) -> Variable {
    let env_borrow = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(env_borrow.clone()),
        kind: OpKind::UnaryOp {
            op: "same_as".to_string(),
            operand: env,
            result_ty: ValueType::Ref(None),
        },
    });
    let tuple_owner = if arg.is_some() {
        format!("Tuple{args_tuple_suffix}")
    } else {
        "Tuple".to_string()
    };
    let args_tuple = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(args_tuple.clone()),
        kind: OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor(&tuple_owner),
            args: Vec::new(),
            result_ty: ValueType::Ref(Some(tuple_owner.clone())),
        },
    });
    if let Some((value, value_ty, class_root)) = arg {
        let value = emit_narrow(graph, block, value, &class_root);
        graph.block_mut(block).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: args_tuple.clone(),
                field: FieldDescriptor {
                    name: "__pos_0".to_string(),
                    owner_root: Some(tuple_owner.clone()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                },
                value: LinkArg::Value(value),
                ty: value_ty,
            },
        });
    }
    let call_result = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(call_result.clone()),
        kind: OpKind::Call {
            target: CallTarget::method("call_mut", Some(call_owner.to_string())),
            args: crate::model::call_args(vec![env_borrow, args_tuple]),
            result_ty,
        },
    });
    call_result
}

fn vec_push_call(result: Variable, out: Variable, item: Variable) -> SpaceOperation {
    SpaceOperation {
        result: Some(result),
        kind: OpKind::Call {
            target: CallTarget::function_path(["vec", "Vec", "push"]),
            args: crate::model::call_args(vec![out, item]),
            result_ty: ValueType::Void,
        },
    }
}

/// Rewrite every recorded `map(it, f).collect()` into
/// `out = Vec::new(); loop { next(it) -> Some(x) => push(out, f(x)); None => break }`.
/// The synthesized `next` is returned so `front::range_iter` then
/// `front::iter_next` can fold it.  Fail-safe: a structural mismatch
/// leaves the residual `collect` (census Skip).
pub(crate) fn rewire_map_collect_sites(
    graph: &mut FunctionGraph,
    sites: &[MapCollectSite],
) -> Vec<(Variable, ValueType)> {
    let mut next_results = Vec::new();
    for site in sites {
        if site.call_once_owner.is_empty() {
            continue;
        }
        match rewire_one_map_collect_site(graph, site) {
            Ok(next_opt) => next_results.push((next_opt, site.inner_item_ty.clone())),
            Err(_decline) => {}
        }
    }
    next_results
}

fn rewire_one_map_collect_site(
    graph: &mut FunctionGraph,
    site: &MapCollectSite,
) -> Result<Variable, String> {
    let name = graph.name.clone();
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: Map::collect result var has no producer block"))?;
    let ci = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .ok_or_else(|| format!("{name}: Map::collect call op not found in block {a}"))?;
    let ops_len = graph.blocks[a].operations.len();
    let (flow_result, _remove_upto) = if ci + 1 == ops_len {
        (site.result_var.clone(), ci)
    } else if ci + 2 == ops_len
        && crate::model::cast_instance_of(
            &graph.blocks[a].operations[ci + 1].kind,
            &site.result_var,
        )
        .is_some()
    {
        let narrowed = graph.blocks[a].operations[ci + 1]
            .result
            .clone()
            .ok_or_else(|| format!("{name}: Map::collect recast has no result"))?;
        (narrowed, ci + 1)
    } else {
        return Err(format!(
            "{name}: Map::collect call is not the last op of block {a}"
        ));
    };
    let mapped = match &graph.blocks[a].operations[ci].kind {
        OpKind::Call { args, .. } if args.len() == 1 => args[0].clone().into_variable(),
        other => {
            return Err(format!(
                "{name}: Map::collect producer op is not a 1-arg call: {other:?}"
            ));
        }
    };
    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: Map::collect call block {a} does not have a single exit"
        ));
    };
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return Err(format!(
            "{name}: Map::collect call block {a} exit is not a plain goto"
        ));
    }
    let saved_exit = exit.clone();
    let b_target = saved_exit.target;
    let ctor = locate_map_ctor(graph, a, &mapped)?;
    if !inner_is_list_or_range(graph, &ctor.inner) {
        return Err(format!(
            "{name}: Map inner iterator is neither a list iter nor an exclusive Range"
        ));
    }
    let emit_block = ctor.block;
    if !var_in_block_scope(graph, emit_block, &ctor.inner)
        || !var_in_block_scope(graph, emit_block, &ctor.env)
    {
        return Err(format!(
            "{name}: map(it, f) operands are not in scope at the constructor"
        ));
    }
    let mut carried: Vec<Variable> = Vec::new();
    for arg in &saved_exit.args {
        if let LinkArg::Value(v) = arg
            && *v != flow_result
            && !carried.contains(v)
        {
            carried.push(v.clone());
        }
    }
    let mut carried_at_emit: Vec<Variable> = Vec::new();
    if emit_block == a {
        carried_at_emit = carried.clone();
    } else {
        for v in &carried {
            // Keep a 1:1 slot with `carried` so `reproduce_exit_args` can
            // index `done` inputs by the same position.  Two A-scope
            // values may back-substitute to one M-scope var; collapsing
            // them here made `map_source` index past `done_inputs`.
            carried_at_emit.push(back_substitute(graph, &[(emit_block, a)], v, &name)?);
        }
    }
    for v in &carried_at_emit {
        if !var_in_block_scope(graph, emit_block, v) {
            return Err(format!(
                "{name}: collect continuation value is not in scope at the map constructor"
            ));
        }
    }
    // The constructor block's exit is redirected at the loop header and
    // block A is left with no predecessor, so any op in A before `collect`
    // would never run. A may hold only the collect call and the trailing
    // recast of its result already accepted above.
    if emit_block != a && ci != 0 {
        return Err(format!(
            "{name}: Map::collect block {a} is orphaned when the constructor \
             lives in block {emit_block}; an earlier op would not run"
        ));
    }

    // --- All structural validation passed; mutate the graph. ---

    let mapped_origin = ctor.mapped.clone();
    let inner = ctor.inner.clone();
    let env = ctor.env.clone();
    let ctor_idx = ctor.op_idx;
    let aggregate = ctor.aggregate;

    if aggregate {
        graph.blocks[emit_block].operations.retain(|op| {
            !matches!(
                &op.kind,
                OpKind::FieldWrite { base, .. } if *base == mapped_origin
            )
        });
    }
    let ctor_idx = graph.blocks[emit_block]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&mapped_origin))
        .unwrap_or(ctor_idx);
    graph.blocks[emit_block].operations.remove(ctor_idx);

    // Drop the collect call and a trailing recast of its result. When
    // the constructor is in an earlier block, A is orphaned; validation
    // already refused every other op in A. An assignment in the
    // constructor's own block is not in A and still runs.
    let ci = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .ok_or_else(|| format!("{name}: Map::collect call vanished before removal"))?;
    let ops_len = graph.blocks[a].operations.len();
    let last = if ci + 1 < ops_len
        && crate::model::cast_instance_of(
            &graph.blocks[a].operations[ci + 1].kind,
            &site.result_var,
        )
        .is_some()
    {
        ci + 1
    } else {
        ci
    };
    for _ in ci..=last {
        graph.blocks[a].operations.remove(ci);
    }

    let out = graph.alloc_value_var();
    graph.blocks[emit_block]
        .operations
        .push(vec_new_call(out.clone()));

    let mut header_sources = vec![inner.clone(), env.clone(), out.clone()];
    for v in &carried_at_emit {
        if !header_sources.contains(v) {
            header_sources.push(v.clone());
        }
    }
    let (header, header_inputs) = graph.create_block_with_arg_vars(header_sources.len());
    let it_h = header_inputs[0].clone();

    let opt = graph.alloc_value_var();
    graph.block_mut(header).operations.push(SpaceOperation {
        result: Some(opt.clone()),
        kind: OpKind::Call {
            target: CallTarget::method("next", None),
            args: crate::model::call_args(vec![it_h]),
            result_ty: ValueType::Ref(None),
        },
    });

    let mut switch_sources = vec![opt.clone()];
    switch_sources.extend(header_inputs.iter().cloned());
    let (switch_bb, switch_inputs) = graph.create_block_with_arg_vars(switch_sources.len());
    let opt_c = switch_inputs[0].clone();
    let it_c = switch_inputs[1].clone();
    let env_c = switch_inputs[2].clone();
    let out_c = switch_inputs[3].clone();
    close_goto_mixed(
        graph,
        header,
        switch_bb,
        switch_sources.iter().cloned().map(LinkArg::Value).collect(),
    );

    let mut body_sources = vec![opt_c.clone(), it_c.clone(), env_c.clone(), out_c.clone()];
    body_sources.extend(switch_inputs.iter().skip(4).cloned());
    let mut done_sources = Vec::with_capacity(1 + carried_at_emit.len());
    for v in &carried_at_emit {
        let idx = header_sources
            .iter()
            .position(|s| s == v)
            .expect("carried map-ctor values are header sources");
        done_sources.push(switch_inputs[1 + idx].clone());
    }
    done_sources.insert(0, out_c.clone());
    let (body_bb, body_inputs) = graph.create_block_with_arg_vars(body_sources.len());
    let (done_bb, done_inputs) = graph.create_block_with_arg_vars(done_sources.len());

    let disc = graph.alloc_value_var();
    graph.block_mut(switch_bb).operations.push(SpaceOperation {
        result: Some(disc.clone()),
        kind: OpKind::FieldRead {
            base: opt_c.clone(),
            field: FieldDescriptor::new("__discriminant", None),
            ty: ValueType::Int,
            pure: true,
        },
    });
    let none_link = Link::new_mixed(
        done_sources.iter().cloned().map(LinkArg::Value).collect(),
        done_bb,
        Some(ExitCase::Const(ConstValue::Int(0))),
    )
    .with_llexitcase_from_exitcase();
    let some_link = Link::new_mixed(
        body_sources.iter().cloned().map(LinkArg::Value).collect(),
        body_bb,
        Some(ExitCase::Const(ConstValue::Int(1))),
    )
    .with_llexitcase_from_exitcase();
    graph.set_control_flow_metadata(
        switch_bb,
        Some(ExitSwitch::Value(disc)),
        vec![none_link, some_link],
    );

    let opt_b = body_inputs[0].clone();
    let it_b = body_inputs[1].clone();
    let env_b = body_inputs[2].clone();
    let out_b = body_inputs[3].clone();
    let payload = graph.alloc_value_var();
    graph.block_mut(body_bb).operations.push(SpaceOperation {
        result: Some(payload.clone()),
        kind: OpKind::FieldRead {
            base: opt_b,
            field: FieldDescriptor::new("__pos_0", None),
            ty: site.payload_ty.clone(),
            pure: true,
        },
    });
    let call_result = emit_call_mut(
        graph,
        body_bb,
        env_b.clone(),
        Some((
            payload,
            site.payload_ty.clone(),
            site.payload_class_root.clone(),
        )),
        &site.call_once_owner,
        site.call_result_ty.clone(),
        &site.args_tuple_suffix,
    );
    let push_result = graph.alloc_value_var();
    graph.block_mut(body_bb).operations.push(vec_push_call(
        push_result,
        out_b.clone(),
        call_result,
    ));
    let mut body_back: Vec<LinkArg> = vec![
        LinkArg::Value(it_b),
        LinkArg::Value(env_b),
        LinkArg::Value(out_b),
    ];
    body_back.extend(body_inputs.iter().skip(4).cloned().map(LinkArg::Value));
    close_goto_mixed(graph, body_bb, header, body_back);

    let out_d = done_inputs[0].clone();
    if carried.len() != done_inputs.len().saturating_sub(1) {
        return Err(format!(
            "{name}: collect continuation arity {} != done extras {}",
            carried.len(),
            done_inputs.len().saturating_sub(1)
        ));
    }
    let done_link_args = reproduce_exit_args(
        &saved_exit,
        &flow_result,
        &out_d,
        &carried,
        &done_inputs[1..],
        &name,
    )?;
    close_goto_mixed(graph, done_bb, b_target, done_link_args);

    let emit_id = graph.blocks[emit_block].id;
    close_goto_mixed(
        graph,
        emit_id,
        header,
        header_sources.into_iter().map(LinkArg::Value).collect(),
    );
    Ok(opt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{ConstValue, Variable};
    use crate::front::iter_next::rewire_next_call_sites;
    use crate::model::{ExitCase, ExitSwitch, Link};

    fn enumerate_next_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "iter", "adapters", "enumerate", "<Impl>", "next"]
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            fun_decl_id: None,
        }
    }

    fn enumerate_ctor_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: [
                "core",
                "iter",
                "traits",
                "iterator",
                "Iterator",
                "enumerate",
            ]
            .iter()
            .map(|s| (*s).to_string())
            .collect(),
            fun_decl_id: None,
        }
    }

    fn iter_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: vec!["core".to_string(), "slice".to_string(), "iter".to_string()],
            fun_decl_id: None,
        }
    }

    fn count_calls(g: &FunctionGraph, pred: impl Fn(&CallTarget) -> bool) -> usize {
        g.blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| matches!(&op.kind, OpKind::Call { target, .. } if pred(target)))
            .count()
    }

    fn build_enumerate_diamond() -> (crate::model::FunctionGraph, Variable, Variable) {
        let mut g = FunctionGraph::new("test_enumerate_next");
        let n = g.startblock;
        let container = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "some".to_string(),
                            "container".to_string(),
                            "make".to_string(),
                        ],
                        fun_decl_id: None,
                    },
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let it = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: iter_target(),
                    args: crate::model::call_args(vec![container]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let enumer = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: enumerate_ctor_target(),
                    args: crate::model::call_args(vec![it]),
                    result_ty: ValueType::Ref(Some("Enumerate".into())),
                },
                true,
            )
            .unwrap();

        let (h, h_args) = g.create_block_with_arg_vars(1);
        let enumer_h = h_args[0].clone();
        let opt = g
            .push_op_var(
                h,
                OpKind::Call {
                    target: enumerate_next_target(),
                    args: crate::model::call_args(vec![enumer_h.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();

        let (c, c_args) = g.create_block_with_arg_vars(2);
        let opt_c = c_args[0].clone();
        let enumer_c = c_args[1].clone();
        let disc = g
            .push_op_var(
                c,
                OpKind::FieldRead {
                    base: opt_c.clone(),
                    field: FieldDescriptor::new("__discriminant", None),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();

        let (some_t, some_args) = g.create_block_with_arg_vars(2);
        let some_opt = some_args[0].clone();
        g.push_op_var(
            some_t,
            OpKind::FieldRead {
                base: some_opt,
                field: FieldDescriptor::new("__pos_0", None),
                ty: ValueType::Ref(None),
                pure: false,
            },
            true,
        );
        g.set_return(some_t, None);

        let (none_t, _none_args) = g.create_block_with_arg_vars(1);
        g.set_return(none_t, None);

        g.set_goto(n, h, vec![enumer.clone()]);
        g.set_goto(h, c, vec![opt.clone(), enumer_h]);
        g.block_mut(c).exitswitch = Some(ExitSwitch::Value(disc));
        g.block_mut(c).exits = vec![
            Link::new_mixed(
                vec![LinkArg::Value(opt_c.clone())],
                none_t,
                Some(ExitCase::Const(ConstValue::Int(0))),
            )
            .with_prevblock(c),
            Link::new_mixed(
                vec![
                    LinkArg::Value(opt_c.clone()),
                    LinkArg::Value(enumer_c.clone()),
                ],
                some_t,
                Some(ExitCase::Const(ConstValue::Int(1))),
            )
            .with_prevblock(c),
        ];
        (g, opt, enumer)
    }

    #[test]
    fn enumerate_next_path_is_recognised() {
        assert!(is_enumerate_next_target(&enumerate_next_target()));
        assert!(is_enumerate_ctor_target(&enumerate_ctor_target()));
        assert!(!is_enumerate_next_target(&iter_target()));
    }

    /// `for (i, x) in xs.iter().enumerate()` lowers to inner `next` + a
    /// count increment + a `(i, x)` tuple — the loop `Enumerate::next`
    /// denotes.  The Opaque `enumerate` / `Enumerate::next` residuals
    /// must not survive.
    #[test]
    fn rewrite_lowers_enumerate_next_to_inner_next_and_count() {
        let (mut g, opt, _enumer) = build_enumerate_diamond();
        let rewritten = rewire_next_call_sites(&mut g, &[(opt.clone(), ValueType::Ref(None))]);
        assert_eq!(rewritten, 1, "the enumerate for-loop must fold");
        assert_eq!(
            count_calls(&g, |t| is_enumerate_ctor_target(t)
                && !matches!(
                    t,
                    CallTarget::SyntheticTransparentCtor { name, .. }
                        if name == ENUMERATE_PAIR_OWNER
                )),
            0,
            "Iterator::enumerate residual must be gone"
        );
        assert_eq!(
            count_calls(&g, is_enumerate_next_target),
            0,
            "Enumerate::next residual must be gone"
        );
        assert_eq!(
            count_calls(&g, |t| matches!(
                t,
                CallTarget::FunctionPath { segments, .. } if segments == &["__iter_next".to_string()]
            )),
            1,
            "native next op on the inner list iterator"
        );
        assert_eq!(
            count_calls(&g, |t| matches!(
                t,
                CallTarget::SyntheticTransparentCtor { name, .. } if name == ENUMERATE_PAIR_OWNER
            )),
            1,
            "enumerate ctor becomes the {{iter, count}} pair"
        );
        let adds = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "add"))
            .count();
        assert_eq!(adds, 1, "count increment on the Some arm");
        assert_eq!(
            count_calls(&g, |t| matches!(
                t,
                CallTarget::SyntheticTransparentCtor { name, .. } if name.starts_with("Tuple<")
            )),
            1,
            "packed (i, item) tuple on the Some arm"
        );
        assert!(
            matches!(
                g.blocks.iter().find(|b| {
                    b.operations.iter().any(|op| {
                        matches!(
                            &op.kind,
                            OpKind::Call {
                                target: CallTarget::FunctionPath { segments, .. },
                                ..
                            } if segments == &["__iter_next".to_string()]
                        )
                    })
                }),
                Some(b) if matches!(b.exitswitch, Some(ExitSwitch::LastException))
            ),
            "inner next closes with StopIteration"
        );
    }

    /// The packed enumerate element is the base iterator's item kind.
    /// A list of ints records `Int`; the tuple's `__pos_1` must not widen
    /// that to `Ref`.
    #[test]
    fn enumerate_tuple_element_keeps_base_item_kind() {
        let (mut g, opt, _enumer) = build_enumerate_diamond();
        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Int)]);
        assert_eq!(rewritten, 1, "the enumerate for-loop must fold");
        let item_tys: Vec<ValueType> = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter_map(|op| match &op.kind {
                OpKind::FieldWrite { field, ty, .. } if field.name == "__pos_1" => Some(ty.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(item_tys, vec![ValueType::Int]);
    }

    /// An Enumerate whose inner iterator is not a list `iter` op stays
    /// residual — packing without the StopIteration diamond would drop
    /// exhaustion.
    #[test]
    fn rewrite_declines_enumerate_over_non_list_inner() {
        let mut g = FunctionGraph::new("test_enumerate_non_list");
        let n = g.startblock;
        let foreign = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["foreign".to_string(), "chars".to_string()],
                        fun_decl_id: None,
                    },
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let enumer = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: enumerate_ctor_target(),
                    args: crate::model::call_args(vec![foreign]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let opt = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: enumerate_next_target(),
                    args: crate::model::call_args(vec![enumer.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(n, Some(opt.clone()));
        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Ref(None))]);
        assert_eq!(rewritten, 0, "non-list inner must decline");
        assert_eq!(
            count_calls(&g, is_enumerate_next_target),
            1,
            "Enumerate::next residual survives a decline"
        );
        assert_eq!(
            count_calls(&g, is_enumerate_ctor_target),
            1,
            "enumerate ctor is not rewritten on a decline"
        );
    }

    fn map_ctor_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "iter", "traits", "iterator", "Iterator", "map"]
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            fun_decl_id: None,
        }
    }

    fn map_collect_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["iter", "adapters", "map", "Map", "collect"]
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            fun_decl_id: None,
        }
    }

    fn collect_site(result_var: Variable) -> MapCollectSite {
        MapCollectSite {
            result_var,
            call_once_owner: "test::closure".into(),
            payload_ty: ValueType::Int,
            payload_class_root: None,
            args_tuple_suffix: String::new(),
            call_result_ty: ValueType::Ref(None),
            inner_item_ty: ValueType::Int,
        }
    }

    /// `map(it, f).collect()` across two blocks — the Call-terminator
    /// shape Charon emits for an opaque `Iterator::map` then `collect`.
    fn build_map_collect_two_blocks() -> (FunctionGraph, Variable) {
        let mut g = FunctionGraph::new("test_map_collect");
        let n = g.startblock;
        let container = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "some".to_string(),
                            "container".to_string(),
                            "make".to_string(),
                        ],
                        fun_decl_id: None,
                    },
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let it = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: iter_target(),
                    args: crate::model::call_args(vec![container]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let env = g.push_op_var(n, OpKind::ConstInt(7), true).unwrap();
        let mapped = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: map_ctor_target(),
                    args: crate::model::call_args(vec![it, env]),
                    result_ty: ValueType::Ref(Some("Map".into())),
                },
                true,
            )
            .unwrap();

        let (a, a_args) = g.create_block_with_arg_vars(1);
        let mapped_a = a_args[0].clone();
        let collected = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: map_collect_target(),
                    args: crate::model::call_args(vec![mapped_a]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(n, a, vec![mapped.clone()]);
        g.set_goto(a, b, vec![collected.clone()]);
        (g, collected)
    }

    #[test]
    fn map_collect_path_is_recognised() {
        assert!(is_map_collect_target(&map_collect_target()));
        assert!(is_map_ctor_target(&map_ctor_target()));
        assert!(is_map_adapter_path("core::iter::adapters::map::Map"));
        assert!(!is_map_collect_target(&iter_target()));
        assert!(!is_map_adapter_path(
            "core::iter::adapters::filter_map::FilterMap"
        ));
    }

    /// The `map` constructor is in the predecessor of the collect block.
    /// An assignment in the collect block would be orphaned when that
    /// predecessor's exit is redirected at the loop header, so the
    /// rewrite declines and the assignment stays reachable from the start.
    #[test]
    fn rewrite_keeps_an_assignment_before_map_collect() {
        let (mut g, collected) = build_map_collect_two_blocks();
        let collect_block = g
            .blocks
            .iter()
            .position(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call { target, .. } if is_map_collect_target(target)
                    )
                })
            })
            .expect("collect block");
        let kept = g.alloc_value_var();
        g.blocks[collect_block].operations.insert(
            0,
            SpaceOperation {
                result: Some(kept),
                kind: OpKind::ConstInt(11),
            },
        );
        let nexts = rewire_map_collect_sites(&mut g, &[collect_site(collected)]);
        assert!(
            nexts.is_empty(),
            "an assignment in the orphaned collect block must decline"
        );
        assert!(
            reachable_from_start(&g).iter().any(|block| {
                g.blocks[*block]
                    .operations
                    .iter()
                    .any(|op| matches!(op.kind, OpKind::ConstInt(11)))
            }),
            "the assignment before Map::collect stays reachable"
        );
        assert_eq!(
            count_calls(&g, is_map_collect_target),
            1,
            "Map::collect residual survives the decline"
        );
    }

    /// `xs.iter().map(f).collect()` becomes `Vec::new` + `next` +
    /// `call_once` + `Vec::push`.  The Opaque map/collect residuals
    /// must not survive, and the inner list iterator must be the
    /// `next` operand so `front::iter_next` can fold it.
    #[test]
    fn rewrite_lowers_map_collect_to_next_call_once_push() {
        let (mut g, collected) = build_map_collect_two_blocks();
        let nexts = rewire_map_collect_sites(&mut g, &[collect_site(collected)]);
        assert_eq!(nexts.len(), 1, "the map.collect chain must fold");
        assert_eq!(
            count_calls(&g, is_map_collect_target),
            0,
            "Map::collect residual must be gone"
        );
        assert_eq!(
            count_calls(&g, is_map_ctor_target),
            0,
            "Iterator::map residual must be gone"
        );
        assert_eq!(
            count_calls(&g, |t| matches!(
                t,
                CallTarget::FunctionPath { segments, .. }
                    if segments == &["vec".to_string(), "Vec".to_string(), "new".to_string()]
            )),
            1,
            "one Vec::new accumulator"
        );
        assert_eq!(
            count_calls(&g, |t| matches!(
                t,
                CallTarget::FunctionPath { segments, .. }
                    if segments == &["vec".to_string(), "Vec".to_string(), "push".to_string()]
            )),
            1,
            "one Vec::push on the Some arm"
        );
        assert_eq!(
            count_calls(
                &g,
                |t| matches!(t, CallTarget::Method { name, .. } if name == "call_mut")
            ),
            1,
            "the Some arm calls the closure through call_mut"
        );
        assert_eq!(
            count_calls(
                &g,
                |t| matches!(t, CallTarget::Method { name, .. } if name == "next")
            ),
            1,
            "synthesized next on the inner iterator"
        );
        let folded = rewire_next_call_sites(&mut g, &nexts);
        assert_eq!(folded, 1, "the synthesized next diamond must fold");
        assert_eq!(
            count_calls(&g, |t| matches!(
                t,
                CallTarget::FunctionPath { segments, .. } if segments == &["__iter_next".to_string()]
            )),
            1,
            "native next op on the inner list iterator"
        );
        assert!(
            matches!(
                g.blocks.iter().find(|b| {
                    b.operations.iter().any(|op| {
                        matches!(
                            &op.kind,
                            OpKind::Call {
                                target: CallTarget::FunctionPath { segments, .. },
                                ..
                            } if segments == &["__iter_next".to_string()]
                        )
                    })
                }),
                Some(b) if matches!(b.exitswitch, Some(ExitSwitch::LastException))
            ),
            "inner next closes with StopIteration"
        );
    }

    /// An adapter whose inner iterator is not a list `iter` and not a
    /// Range stays residual — synthesizing `next` without the
    /// StopIteration diamond would drop exhaustion.
    #[test]
    fn rewrite_declines_map_collect_over_non_list_inner() {
        let mut g = FunctionGraph::new("test_map_collect_non_list");
        let n = g.startblock;
        let foreign = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["foreign".to_string(), "chars".to_string()],
                        fun_decl_id: None,
                    },
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let env = g.push_op_var(n, OpKind::ConstInt(7), true).unwrap();
        let mapped = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: map_ctor_target(),
                    args: crate::model::call_args(vec![foreign, env]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (a, a_args) = g.create_block_with_arg_vars(1);
        let mapped_a = a_args[0].clone();
        let collected = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: map_collect_target(),
                    args: crate::model::call_args(vec![mapped_a]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_goto(n, a, vec![mapped]);
        g.set_return(a, Some(collected.clone()));
        let nexts = rewire_map_collect_sites(&mut g, &[collect_site(collected)]);
        assert!(nexts.is_empty(), "non-list inner must decline");
        assert_eq!(
            count_calls(&g, is_map_collect_target),
            1,
            "Map::collect residual survives a decline"
        );
        assert_eq!(
            count_calls(&g, is_map_ctor_target),
            1,
            "Iterator::map residual survives a decline"
        );
    }

    /// Empty `call_once_owner` means the capture could not name a
    /// concrete closure ADT.  The rewrite must not guess a callee.
    #[test]
    fn rewrite_declines_when_closure_env_is_not_a_concrete_adt() {
        let (mut g, collected) = build_map_collect_two_blocks();
        let mut site = collect_site(collected);
        site.call_once_owner.clear();
        let nexts = rewire_map_collect_sites(&mut g, &[site]);
        assert!(nexts.is_empty(), "missing call_once owner must decline");
        assert_eq!(
            count_calls(&g, is_map_collect_target),
            1,
            "Map::collect residual survives a missing closure ADT"
        );
    }

    /// The collect continuation may forward several live values besides
    /// the Vec.  Each A-scope slot must keep a matching `done` input —
    /// collapsing duplicate M-scope images used to index past that vec
    /// inside `reproduce_exit_args`.
    #[test]
    fn rewrite_threads_extra_live_values_to_the_continuation() {
        let mut g = FunctionGraph::new("test_map_collect_carried");
        let n = g.startblock;
        let container = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "some".to_string(),
                            "container".to_string(),
                            "make".to_string(),
                        ],
                        fun_decl_id: None,
                    },
                    args: Vec::new(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let it = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: iter_target(),
                    args: crate::model::call_args(vec![container]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let env = g.push_op_var(n, OpKind::ConstInt(7), true).unwrap();
        let extra = g.push_op_var(n, OpKind::ConstInt(9), true).unwrap();
        let mapped = g
            .push_op_var(
                n,
                OpKind::Call {
                    target: map_ctor_target(),
                    args: crate::model::call_args(vec![it, env]),
                    result_ty: ValueType::Ref(Some("Map".into())),
                },
                true,
            )
            .unwrap();

        let (a, a_args) = g.create_block_with_arg_vars(2);
        let mapped_a = a_args[0].clone();
        let extra_a = a_args[1].clone();
        let collected = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: map_collect_target(),
                    args: crate::model::call_args(vec![mapped_a]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(2);
        g.set_return(b, None);
        g.set_goto(n, a, vec![mapped, extra]);
        g.set_goto(a, b, vec![collected.clone(), extra_a]);
        let nexts = rewire_map_collect_sites(&mut g, &[collect_site(collected)]);
        assert_eq!(
            nexts.len(),
            1,
            "extra live values must not decline the rewrite"
        );
        assert_eq!(
            count_calls(&g, is_map_collect_target),
            0,
            "Map::collect residual must be gone"
        );
    }

    fn reachable_from_start(g: &FunctionGraph) -> Vec<usize> {
        let mut seen = vec![false; g.blocks.len()];
        let mut stack = vec![g.startblock.0];
        while let Some(block) = stack.pop() {
            if seen.get(block).copied().unwrap_or(true) {
                continue;
            }
            seen[block] = true;
            for link in &g.blocks[block].exits {
                stack.push(link.target.0);
            }
        }
        seen.iter()
            .enumerate()
            .filter_map(|(i, on)| on.then_some(i))
            .collect()
    }

    /// `for _ in xs.iter().enumerate()` never reads the Some payload.
    /// The rewrite still folds, and `__pos_1` carries the inner item type.
    #[test]
    fn rewrite_enumerate_unread_payload_packs_item_type() {
        let (mut g, opt, _) = build_enumerate_diamond();
        let some = g
            .blocks
            .iter()
            .position(|block| {
                block.operations.iter().any(|op| {
                    matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__pos_0")
                })
            })
            .expect("some arm");
        let some_id = g.blocks[some].id;
        g.blocks[some].operations.clear();
        g.set_return(some_id, None);
        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Int)]);
        assert_eq!(rewritten, 1, "an unread Some payload still folds");
        assert!(
            g.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                matches!(
                    &op.kind,
                    OpKind::FieldWrite { field, ty, .. }
                        if field.name == "__pos_1"
                            && field.owner_root.as_deref() == Some("Tuple<usize,isize>")
                            && *ty == ValueType::Int
                )
            }),
            "__pos_1 is the inner item type on a Tuple"
        );
        assert_eq!(
            count_calls(&g, is_enumerate_next_target),
            0,
            "Enumerate::next residual must be gone"
        );
    }

    /// Two enumerate sites with different item types do not share a tuple owner.
    #[test]
    fn two_enumerate_item_types_keep_distinct_owners() {
        let mut g = FunctionGraph::new("two_enumerate_items");
        let (b0, a0) = g.create_block_with_arg_vars(1);
        let (b1, a1) = g.create_block_with_arg_vars(1);
        let pair0 = g.alloc_value_var();
        let pair1 = g.alloc_value_var();
        pack_enumerate_payload(&mut g, b0.0, &a0[0], &ValueType::Int, &pair0, "int_site").unwrap();
        pack_enumerate_payload(&mut g, b1.0, &a1[0], &ValueType::Str, &pair1, "str_site").unwrap();
        let owners: Vec<String> = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter_map(|op| match &op.kind {
                OpKind::FieldWrite { field, .. } if field.name == "__pos_1" => {
                    field.owner_root.clone()
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            owners,
            vec![
                "Tuple<usize,isize>".to_string(),
                "Tuple<usize,String>".to_string()
            ]
        );
        let ctors: Vec<String> = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { name, .. },
                    ..
                } if name.starts_with("Tuple<") => Some(name.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(ctors, owners);
    }

    /// A successor that reads `__pos_0` of the forwarded payload receives
    /// the packed tuple, not the pre-collapse shape.
    #[test]
    fn successor_reads_pos0_of_forwarded_packed_payload() {
        let (mut g, opt, _) = build_enumerate_diamond();
        let some = g
            .blocks
            .iter()
            .position(|block| {
                block.operations.iter().any(|op| {
                    matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__pos_0")
                })
            })
            .expect("some arm");
        let carrier = g.blocks[some].inputargs[0].clone();
        let some_id = g.blocks[some].id;
        let (succ, succ_args) = g.create_block_with_arg_vars(1);
        let received = succ_args[0].clone();
        g.push_op_var(
            succ,
            OpKind::FieldRead {
                base: received.clone(),
                field: FieldDescriptor::new("__pos_0", None),
                ty: ValueType::Unsigned,
                pure: false,
            },
            true,
        )
        .unwrap();
        g.set_return(succ, None);
        g.block_mut(some_id).exits = vec![Link::new_mixed(
            vec![LinkArg::Value(carrier.clone())],
            succ,
            None,
        )
        .with_prevblock(some_id)];
        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Str)]);
        assert_eq!(rewritten, 1, "a successor __pos_0 read of the forward folds");
        let exit_val = match &g.block_mut(some_id).exits[0].args[0] {
            LinkArg::Value(v) => v.clone(),
            other => panic!("packed forward must be a value, got {other:?}"),
        };
        assert_ne!(exit_val, carrier, "the successor must not receive the old payload");
        let succ_block = g.blocks.iter().find(|b| b.id == succ).expect("successor");
        assert!(
            succ_block.operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::FieldRead { base, field, .. }
                        if base == &received
                            && field.name == "__pos_0"
                            && field.owner_root.as_deref() == Some("Tuple<usize,String>")
                )
            }),
            "the successor __pos_0 read is the packed tuple owner"
        );
    }

    /// A merge successor has another predecessor that still passes the old
    /// payload. Painting its `__pos_N` reads would retarget that edge too.
    #[test]
    fn rewrite_declines_forward_into_merge_successor() {
        let (mut g, opt, _) = build_enumerate_diamond();
        let some = g
            .blocks
            .iter()
            .position(|block| {
                block.operations.iter().any(|op| {
                    matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__pos_0")
                })
            })
            .expect("some arm");
        let carrier = g.blocks[some].inputargs[0].clone();
        let some_id = g.blocks[some].id;
        let (succ, succ_args) = g.create_block_with_arg_vars(1);
        g.push_op_var(
            succ,
            OpKind::FieldRead {
                base: succ_args[0].clone(),
                field: FieldDescriptor::new("__pos_0", None),
                ty: ValueType::Unsigned,
                pure: false,
            },
            true,
        )
        .unwrap();
        g.set_return(succ, None);
        let (other, _) = g.create_block_with_arg_vars(0);
        let other_val = g.alloc_value_var();
        g.block_mut(other).exits = vec![Link::new_mixed(
            vec![LinkArg::Value(other_val)],
            succ,
            None,
        )
        .with_prevblock(other)];
        g.block_mut(some_id).exits = vec![Link::new_mixed(
            vec![LinkArg::Value(carrier)],
            succ,
            None,
        )
        .with_prevblock(some_id)];
        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Str)]);
        assert_eq!(rewritten, 0, "a merge successor must decline");
        assert_eq!(count_calls(&g, is_enumerate_ctor_target), 1);
    }

    /// Switching on the forwarded slot is a use of the old payload shape.
    #[test]
    fn rewrite_declines_successor_switching_on_forwarded_slot() {
        let (mut g, opt, _) = build_enumerate_diamond();
        let some = g
            .blocks
            .iter()
            .position(|block| {
                block.operations.iter().any(|op| {
                    matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__pos_0")
                })
            })
            .expect("some arm");
        let carrier = g.blocks[some].inputargs[0].clone();
        let some_id = g.blocks[some].id;
        let (succ, succ_args) = g.create_block_with_arg_vars(1);
        let received = succ_args[0].clone();
        g.push_op_var(
            succ,
            OpKind::FieldRead {
                base: received.clone(),
                field: FieldDescriptor::new("__pos_0", None),
                ty: ValueType::Unsigned,
                pure: false,
            },
            true,
        )
        .unwrap();
        g.set_return(succ, None);
        g.block_mut(succ).exitswitch = Some(ExitSwitch::Value(received));
        g.block_mut(some_id).exits = vec![Link::new_mixed(
            vec![LinkArg::Value(carrier)],
            succ,
            None,
        )
        .with_prevblock(some_id)];
        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Str)]);
        assert_eq!(rewritten, 0, "an exitswitch on the forwarded slot must decline");
        assert_eq!(count_calls(&g, is_enumerate_ctor_target), 1);
    }

    /// A Some arm that uses the payload slot for something other than
    /// `__pos_0` declines before the enumerate constructor is rewritten.
    #[test]
    fn rewrite_declines_enumerate_payload_used_outside_pos0() {
        let (mut g, opt, _) = build_enumerate_diamond();
        for block in &mut g.blocks {
            for op in &mut block.operations {
                if let OpKind::FieldRead { base, field, .. } = &op.kind
                    && field.name == "__pos_0"
                {
                    let base = base.clone();
                    op.kind = OpKind::Call {
                        target: CallTarget::function_path(["uses", "payload"]),
                        args: crate::model::call_args(vec![base]),
                        result_ty: ValueType::Ref(None),
                    };
                }
            }
        }
        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Ref(None))]);
        assert_eq!(rewritten, 0, "a non-__pos_0 payload use must decline");
        assert_eq!(
            count_calls(&g, is_enumerate_ctor_target),
            1,
            "enumerate ctor is not rewritten on a decline"
        );
        assert_eq!(
            count_calls(&g, |t| matches!(
                t,
                CallTarget::SyntheticTransparentCtor { name, .. } if name == ENUMERATE_PAIR_OWNER
            )),
            0,
            "the pair ctor is not installed on a decline"
        );
    }

    /// A `__pos_0` read plus a forward of the same carrier is still a
    /// second reference. The rewrite must decline before the constructor
    /// is replaced.
    #[test]
    fn rewrite_declines_enumerate_payload_forwarded_beside_pos0() {
        let (mut g, opt, _) = build_enumerate_diamond();
        let some = g
            .blocks
            .iter()
            .position(|block| {
                block.operations.iter().any(|op| {
                    matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == "__pos_0")
                })
            })
            .expect("some arm");
        let carrier = g.blocks[some].inputargs[0].clone();
        g.blocks[some].exits[0].args.push(LinkArg::Value(carrier));
        let rewritten = rewire_next_call_sites(&mut g, &[(opt, ValueType::Ref(None))]);
        assert_eq!(
            rewritten, 0,
            "forwarding the carrier beside __pos_0 must decline"
        );
        assert_eq!(
            count_calls(&g, is_enumerate_ctor_target),
            1,
            "enumerate ctor is not rewritten on a decline"
        );
        assert_eq!(
            count_calls(&g, |t| matches!(
                t,
                CallTarget::SyntheticTransparentCtor { name, .. } if name == ENUMERATE_PAIR_OWNER
            )),
            0,
            "the pair ctor is not installed on a decline"
        );
    }

    /// A trailing recast is accepted only when it recasts the collect result.
    #[test]
    fn rewrite_declines_recast_of_a_different_value_after_collect() {
        let (mut g, collected) = build_map_collect_two_blocks();
        let collect_block = g
            .blocks
            .iter()
            .position(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call { target, .. } if is_map_collect_target(target)
                    )
                })
            })
            .expect("collect block");
        let other = g.alloc_value_var();
        let narrowed = g.alloc_value_var();
        g.blocks[collect_block].operations.push(SpaceOperation {
            result: Some(narrowed),
            kind: OpKind::Call {
                target: CallTarget::function_path(["__cast_instance_intrinsic"]),
                args: vec![
                    LinkArg::from(other),
                    LinkArg::from(ConstValue::byte_str("Vec")),
                ],
                result_ty: ValueType::Ref(Some("Vec".into())),
            },
        });
        let nexts = rewire_map_collect_sites(&mut g, &[collect_site(collected)]);
        assert!(
            nexts.is_empty(),
            "a recast of some other value must decline"
        );
        assert_eq!(
            count_calls(&g, is_map_collect_target),
            1,
            "Map::collect residual survives a foreign recast"
        );
    }

    /// `(0..n).map(move |_| owned.len()).collect()` reuses one closure.
    /// `FnOnce::call_once` moves the env; the header then reads that moved
    /// value. `FnMut::call_mut` takes a reborrow (`same_as` of the
    /// loop-carried env) and the back edge carries the original env.
    #[test]
    fn map_collect_calls_closure_through_call_mut_reborrow() {
        let (mut g, collected) = build_map_collect_two_blocks();
        let nexts = rewire_map_collect_sites(&mut g, &[collect_site(collected)]);
        assert_eq!(nexts.len(), 1, "the map.collect chain must fold");
        let (name, args) = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .find_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::Method { name, .. },
                    args,
                    ..
                } if name == "call_once" || name == "call_mut" => {
                    Some((name.clone(), args.clone()))
                }
                _ => None,
            })
            .expect("the loop body must call the closure");
        assert_eq!(
            name, "call_mut",
            "the closure is FnMut, reused each iteration"
        );
        let receiver = args[0].clone().into_variable();
        let env = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .find_map(|op| match (&op.kind, &op.result) {
                (OpKind::UnaryOp { op, operand, .. }, Some(result))
                    if op == "same_as" && result == &receiver =>
                {
                    Some(operand.clone())
                }
                _ => None,
            })
            .expect("call_mut receiver must be a reborrow of the loop env");
        assert_ne!(receiver, env, "the reborrow is not the loop-carried env");
        let threads_original = g.blocks.iter().any(|b| {
            b.exits.iter().any(|link| {
                link.args
                    .iter()
                    .any(|arg| matches!(arg, LinkArg::Value(v) if v == &env))
                    && link
                        .args
                        .iter()
                        .all(|arg| !matches!(arg, LinkArg::Value(v) if v == &receiver))
            })
        });
        assert!(
            threads_original,
            "the loop header must carry the original env, not the reborrow"
        );
    }
}
