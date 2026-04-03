//! Graph-based jtransform: semantic rewrite pass.
//!
//! RPython equivalent: jtransform.py Transformer.optimize_block()
//!
//! Transforms a MajitGraph by rewriting operations:
//! - FieldRead on virtualizable fields → VableFieldRead marker
//! - FieldWrite on virtualizable fields → VableFieldWrite marker
//! - ArrayRead on virtualizable arrays → VableArrayRead marker
//! - Call classification → elidable/residual/may_force tagging

use serde::{Deserialize, Serialize};

use crate::call_match::CallDescriptor;
use crate::graph::{
    BasicBlockId, CallTarget, FieldDescriptor, MajitGraph, Op, OpKind, Terminator, ValueId,
    ValueType,
};
use majit_ir::descr::{EffectInfo, ExtraEffect, OopSpecIndex};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VirtualizableFieldDescriptor {
    pub name: String,
    pub owner_root: Option<String>,
    pub index: usize,
}

impl VirtualizableFieldDescriptor {
    pub fn new(name: impl Into<String>, owner_root: Option<String>, index: usize) -> Self {
        Self {
            name: name.into(),
            owner_root,
            index,
        }
    }

    fn matches(&self, field: &FieldDescriptor) -> bool {
        self.name == field.name
            && self
                .owner_root
                .as_ref()
                .is_none_or(|owner| field.owner_root.as_ref() == Some(owner))
    }
}

/// Configuration for the graph rewrite pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTransformConfig {
    /// Whether to rewrite virtualizable field/array accesses.
    pub lower_virtualizable: bool,
    /// Whether to classify function calls by effect.
    pub classify_calls: bool,
    /// Virtualizable scalar field descriptors.
    #[serde(default)]
    pub vable_fields: Vec<VirtualizableFieldDescriptor>,
    /// Virtualizable array field descriptors.
    #[serde(default)]
    pub vable_arrays: Vec<VirtualizableFieldDescriptor>,
    /// Explicit call effect overrides.
    ///
    /// RPython equivalent: effect classification travels on call descriptors
    /// rather than being rediscovered from source text.
    #[serde(default)]
    pub call_effects: Vec<CallEffectOverride>,
}

impl Default for GraphTransformConfig {
    fn default() -> Self {
        Self {
            lower_virtualizable: true,
            classify_calls: true,
            vable_fields: Vec::new(),
            vable_arrays: Vec::new(),
            call_effects: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CallEffectKind {
    Elidable,
    Residual,
    MayForce,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallEffectOverride {
    pub descriptor: CallDescriptor,
}

impl CallEffectOverride {
    pub fn new(target: CallTarget, effect: CallEffectKind) -> Self {
        Self {
            descriptor: CallDescriptor::override_effect(target, effect_info_for_kind(effect)),
        }
    }
}

fn effect_info_for_kind(effect: CallEffectKind) -> EffectInfo {
    match effect {
        CallEffectKind::Elidable => EffectInfo::elidable(),
        CallEffectKind::Residual => EffectInfo::new(ExtraEffect::CanRaise, OopSpecIndex::None),
        CallEffectKind::MayForce => EffectInfo::new(
            ExtraEffect::ForcesVirtualOrVirtualizable,
            OopSpecIndex::None,
        ),
    }
}

/// A note about a transformation applied to the graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphTransformNote {
    pub function: String,
    pub detail: String,
}

/// Result of a graph transformation pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTransformResult {
    pub graph: MajitGraph,
    pub notes: Vec<GraphTransformNote>,
    /// Number of ops rewritten by virtualizable lowering.
    pub vable_rewrites: usize,
    /// Number of calls classified.
    pub calls_classified: usize,
}

/// Rewrite a semantic graph with JIT-specific transformations.
///
/// RPython equivalent: jtransform.py `Transformer.optimize_block()`
///
/// This is the graph-based replacement for the old TracePattern →
/// LoweringRecipe string-matching pipeline.
pub fn rewrite_graph(graph: &MajitGraph, config: &GraphTransformConfig) -> GraphTransformResult {
    let mut notes = Vec::new();
    let mut rewritten = graph.clone();
    let mut vable_rewrites = 0usize;
    let mut calls_classified = 0usize;
    let mut aliases: std::collections::HashMap<ValueId, ValueId> = std::collections::HashMap::new();

    // Build lookup sets for virtualizable fields/arrays
    // Track which ValueIds are results of reading a virtualizable array field.
    // RPython jtransform.py tracks vable_array_vars for this purpose.
    let mut vable_array_values: std::collections::HashMap<crate::graph::ValueId, usize> =
        std::collections::HashMap::new();

    for block in &mut rewritten.blocks {
        let mut new_ops = Vec::with_capacity(block.ops.len());

        for original_op in &block.ops {
            let op = remap_op(original_op, &aliases);
            match &op.kind {
                // ── hint(access_directly=True) / hint(fresh_virtualizable=True) ──
                // RPython jtransform.py:655 — consume as identity during translation.
                OpKind::Call { target, args, .. } if is_vable_identity_hint(target) => {
                    notes.push(GraphTransformNote {
                        function: graph.name.clone(),
                        detail: format!("rewrite: {target}(...) → identity"),
                    });
                    if let (Some(result), Some(arg)) = (op.result, args.first().copied()) {
                        aliases.insert(result, resolve_alias(arg, &aliases));
                    }
                    continue;
                }

                // ── hint(force_virtualizable=True) ──
                // RPython jtransform.py:650 — emit hint_force_virtualizable op,
                // preserving the value as an identity result.
                OpKind::Call { target, args, .. } if is_vable_force_hint(target) => {
                    notes.push(GraphTransformNote {
                        function: graph.name.clone(),
                        detail: format!("rewrite: {target}(...) → VableForce"),
                    });
                    vable_rewrites += 1;
                    if let (Some(result), Some(arg)) = (op.result, args.first().copied()) {
                        aliases.insert(result, resolve_alias(arg, &aliases));
                    }
                    new_ops.push(Op {
                        result: None,
                        kind: OpKind::VableForce,
                    });
                    continue;
                }

                // ── Virtualizable field read → VableFieldRead ──
                // RPython jtransform.py:832 `rewrite_op_getfield`
                OpKind::FieldRead { field, ty, .. } if config.lower_virtualizable => {
                    // Track if this field read is on a virtualizable array
                    if let Some(array_field) = config
                        .vable_arrays
                        .iter()
                        .find(|candidate| candidate.matches(field))
                    {
                        if let Some(result) = op.result {
                            vable_array_values.insert(result, array_field.index);
                        }
                    }
                    if let Some(vable_field) = config
                        .vable_fields
                        .iter()
                        .find(|candidate| candidate.matches(field))
                    {
                        notes.push(GraphTransformNote {
                            function: graph.name.clone(),
                            detail: format!(
                                "rewrite: {} → VableFieldRead[{}]",
                                field.name, vable_field.index
                            ),
                        });
                        vable_rewrites += 1;
                        new_ops.push(Op {
                            result: op.result,
                            kind: OpKind::VableFieldRead {
                                field_index: vable_field.index,
                                ty: ty.clone(),
                            },
                        });
                        continue;
                    }
                }

                // ── Virtualizable field write → VableFieldWrite ──
                // RPython jtransform.py:923 `_rewrite_op_setfield`
                OpKind::FieldWrite {
                    field, value, ty, ..
                } if config.lower_virtualizable => {
                    if let Some(vable_field) = config
                        .vable_fields
                        .iter()
                        .find(|candidate| candidate.matches(field))
                    {
                        notes.push(GraphTransformNote {
                            function: graph.name.clone(),
                            detail: format!(
                                "rewrite: {} = ... → VableFieldWrite[{}]",
                                field.name, vable_field.index
                            ),
                        });
                        vable_rewrites += 1;
                        new_ops.push(Op {
                            result: op.result,
                            kind: OpKind::VableFieldWrite {
                                field_index: vable_field.index,
                                value: *value,
                                ty: ty.clone(),
                            },
                        });
                        continue;
                    }
                }

                // ── Virtualizable array read → VableArrayRead ──
                // RPython jtransform.py:760 `getarrayitem_vable`
                OpKind::ArrayRead {
                    base,
                    index,
                    item_ty,
                } if config.lower_virtualizable => {
                    if let Some(&arr_idx) = vable_array_values.get(base) {
                        notes.push(GraphTransformNote {
                            function: graph.name.clone(),
                            detail: format!("rewrite: array[idx] → VableArrayRead[{arr_idx}]"),
                        });
                        vable_rewrites += 1;
                        new_ops.push(Op {
                            result: op.result,
                            kind: OpKind::VableArrayRead {
                                array_index: arr_idx,
                                elem_index: *index,
                                item_ty: item_ty.clone(),
                            },
                        });
                        continue;
                    }
                }

                // ── Virtualizable array write → VableArrayWrite ──
                // RPython jtransform.py:794 `setarrayitem_vable`
                OpKind::ArrayWrite {
                    base,
                    index,
                    value,
                    item_ty,
                } if config.lower_virtualizable => {
                    if let Some(&arr_idx) = vable_array_values.get(base) {
                        notes.push(GraphTransformNote {
                            function: graph.name.clone(),
                            detail: format!("rewrite: array[idx] = v → VableArrayWrite[{arr_idx}]"),
                        });
                        vable_rewrites += 1;
                        new_ops.push(Op {
                            result: op.result,
                            kind: OpKind::VableArrayWrite {
                                array_index: arr_idx,
                                elem_index: *index,
                                value: *value,
                                item_ty: item_ty.clone(),
                            },
                        });
                        continue;
                    }
                }

                // ── Call classification → rewrite to typed call ──
                // RPython jtransform.py: classify calls by effect info
                OpKind::Call {
                    target,
                    args,
                    result_ty,
                } if config.classify_calls => {
                    if let Some((descriptor, effect)) = classify_call(target, &config.call_effects)
                    {
                        notes.push(GraphTransformNote {
                            function: graph.name.clone(),
                            detail: format!("call {target} → {}", effect.as_str()),
                        });
                        calls_classified += 1;
                        let rewritten_kind = match effect {
                            CallEffectKind::Elidable => OpKind::CallElidable {
                                descriptor,
                                args: args.clone(),
                                result_ty: result_ty.clone(),
                            },
                            CallEffectKind::Residual => OpKind::CallResidual {
                                descriptor,
                                args: args.clone(),
                                result_ty: result_ty.clone(),
                            },
                            CallEffectKind::MayForce => OpKind::CallMayForce {
                                descriptor,
                                args: args.clone(),
                                result_ty: result_ty.clone(),
                            },
                        };
                        new_ops.push(Op {
                            result: op.result,
                            kind: rewritten_kind,
                        });
                        continue;
                    }
                }

                // ── Unknown ops ──
                OpKind::Unknown { kind } => {
                    notes.push(GraphTransformNote {
                        function: graph.name.clone(),
                        detail: format!("unknown op: {:?}", kind),
                    });
                }

                _ => {}
            }
            new_ops.push(op);
        }

        block.ops = new_ops;

        block.terminator = remap_terminator(&block.terminator, &aliases);

        if let Terminator::Abort { reason } = &block.terminator {
            notes.push(GraphTransformNote {
                function: graph.name.clone(),
                detail: format!("abort: {reason}"),
            });
        }
    }

    GraphTransformResult {
        graph: rewritten,
        notes,
        vable_rewrites,
        calls_classified,
    }
}

fn resolve_alias(
    mut value: ValueId,
    aliases: &std::collections::HashMap<ValueId, ValueId>,
) -> ValueId {
    while let Some(next) = aliases.get(&value).copied() {
        if next == value {
            break;
        }
        value = next;
    }
    value
}

fn remap_value(value: ValueId, aliases: &std::collections::HashMap<ValueId, ValueId>) -> ValueId {
    resolve_alias(value, aliases)
}

fn remap_op(op: &Op, aliases: &std::collections::HashMap<ValueId, ValueId>) -> Op {
    let kind = match &op.kind {
        OpKind::Input { .. }
        | OpKind::ConstInt(_)
        | OpKind::VableForce
        | OpKind::Unknown { .. } => op.kind.clone(),
        OpKind::FieldRead { base, field, ty } => OpKind::FieldRead {
            base: remap_value(*base, aliases),
            field: field.clone(),
            ty: ty.clone(),
        },
        OpKind::FieldWrite {
            base,
            field,
            value,
            ty,
        } => OpKind::FieldWrite {
            base: remap_value(*base, aliases),
            field: field.clone(),
            value: remap_value(*value, aliases),
            ty: ty.clone(),
        },
        OpKind::ArrayRead {
            base,
            index,
            item_ty,
        } => OpKind::ArrayRead {
            base: remap_value(*base, aliases),
            index: remap_value(*index, aliases),
            item_ty: item_ty.clone(),
        },
        OpKind::ArrayWrite {
            base,
            index,
            value,
            item_ty,
        } => OpKind::ArrayWrite {
            base: remap_value(*base, aliases),
            index: remap_value(*index, aliases),
            value: remap_value(*value, aliases),
            item_ty: item_ty.clone(),
        },
        OpKind::Call {
            target,
            args,
            result_ty,
        } => OpKind::Call {
            target: target.clone(),
            args: args
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            result_ty: result_ty.clone(),
        },
        OpKind::GuardTrue { cond } => OpKind::GuardTrue {
            cond: remap_value(*cond, aliases),
        },
        OpKind::GuardFalse { cond } => OpKind::GuardFalse {
            cond: remap_value(*cond, aliases),
        },
        OpKind::VableFieldRead { .. } => op.kind.clone(),
        OpKind::VableFieldWrite {
            field_index,
            value,
            ty,
        } => OpKind::VableFieldWrite {
            field_index: *field_index,
            value: remap_value(*value, aliases),
            ty: ty.clone(),
        },
        OpKind::VableArrayRead {
            array_index,
            elem_index,
            item_ty,
        } => OpKind::VableArrayRead {
            array_index: *array_index,
            elem_index: remap_value(*elem_index, aliases),
            item_ty: item_ty.clone(),
        },
        OpKind::VableArrayWrite {
            array_index,
            elem_index,
            value,
            item_ty,
        } => OpKind::VableArrayWrite {
            array_index: *array_index,
            elem_index: remap_value(*elem_index, aliases),
            value: remap_value(*value, aliases),
            item_ty: item_ty.clone(),
        },
        OpKind::BinOp {
            op,
            lhs,
            rhs,
            result_ty,
        } => OpKind::BinOp {
            op: op.clone(),
            lhs: remap_value(*lhs, aliases),
            rhs: remap_value(*rhs, aliases),
            result_ty: result_ty.clone(),
        },
        OpKind::UnaryOp {
            op,
            operand,
            result_ty,
        } => OpKind::UnaryOp {
            op: op.clone(),
            operand: remap_value(*operand, aliases),
            result_ty: result_ty.clone(),
        },
        OpKind::CallElidable {
            descriptor,
            args,
            result_ty,
        } => OpKind::CallElidable {
            descriptor: descriptor.clone(),
            args: args
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            result_ty: result_ty.clone(),
        },
        OpKind::CallResidual {
            descriptor,
            args,
            result_ty,
        } => OpKind::CallResidual {
            descriptor: descriptor.clone(),
            args: args
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            result_ty: result_ty.clone(),
        },
        OpKind::CallMayForce {
            descriptor,
            args,
            result_ty,
        } => OpKind::CallMayForce {
            descriptor: descriptor.clone(),
            args: args
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            result_ty: result_ty.clone(),
        },
    };
    Op {
        result: op.result,
        kind,
    }
}

fn remap_terminator(
    term: &Terminator,
    aliases: &std::collections::HashMap<ValueId, ValueId>,
) -> Terminator {
    match term {
        Terminator::Goto { target, args } => Terminator::Goto {
            target: *target,
            args: args
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
        },
        Terminator::Branch {
            cond,
            if_true,
            true_args,
            if_false,
            false_args,
        } => Terminator::Branch {
            cond: remap_value(*cond, aliases),
            if_true: *if_true,
            true_args: true_args
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            if_false: *if_false,
            false_args: false_args
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
        },
        Terminator::Return(val) => Terminator::Return(val.map(|v| remap_value(v, aliases))),
        Terminator::Abort { reason } => Terminator::Abort {
            reason: reason.clone(),
        },
        Terminator::Unreachable => Terminator::Unreachable,
    }
}

fn classify_vable_hint(target: &CallTarget) -> Option<crate::hints::VirtualizableHintKind> {
    target
        .path_segments()
        .and_then(|segments| crate::hints::classify_virtualizable_hint_segments(segments))
}

fn is_vable_identity_hint(target: &CallTarget) -> bool {
    matches!(
        classify_vable_hint(target),
        Some(
            crate::hints::VirtualizableHintKind::AccessDirectly
                | crate::hints::VirtualizableHintKind::FreshVirtualizable
        )
    )
}

fn is_vable_force_hint(target: &CallTarget) -> bool {
    matches!(
        classify_vable_hint(target),
        Some(crate::hints::VirtualizableHintKind::ForceVirtualizable)
    )
}

impl CallEffectKind {
    fn as_str(self) -> &'static str {
        match self {
            CallEffectKind::Elidable => "elidable",
            CallEffectKind::Residual => "residual",
            CallEffectKind::MayForce => "may_force",
        }
    }
}

/// Match CallTarget loosely: generic receivers (lowercase like "handler",
/// "self") match any pattern receiver type.
fn call_target_matches_loose(pattern: &CallTarget, target: &CallTarget) -> bool {
    match (pattern, target) {
        (
            CallTarget::Method {
                name: pn,
                receiver_root: pr,
            },
            CallTarget::Method {
                name: tn,
                receiver_root: tr,
            },
        ) => {
            if pn != tn {
                return false;
            }
            match (pr.as_deref(), tr.as_deref()) {
                (Some(p), Some(t)) => p == t || crate::call_match::is_generic_receiver(t),
                _ => true,
            }
        }
        _ => pattern == target,
    }
}

/// Classify a call's side-effect level.
///
/// RPython equivalent: jtransform.py effect classification
/// (EF_ELIDABLE, EF_FORCES_VIRTUAL, etc.)
fn classify_call(
    target: &CallTarget,
    overrides: &[CallEffectOverride],
) -> Option<(CallDescriptor, CallEffectKind)> {
    fn classify_effect_info(info: &majit_ir::descr::EffectInfo) -> CallEffectKind {
        if info.forces_virtual_or_virtualizable() {
            CallEffectKind::MayForce
        } else if info.is_elidable() {
            CallEffectKind::Elidable
        } else {
            CallEffectKind::Residual
        }
    }

    if let Some(descriptor) = overrides
        .iter()
        .find(|override_| call_target_matches_loose(&override_.descriptor.target, target))
        .map(|override_| override_.descriptor.clone())
    {
        let effect = classify_effect_info(&descriptor.effect_info());
        return Some((descriptor, effect));
    }
    let descriptor = crate::call_match::describe_call(target)?;
    let effect = classify_effect_info(&descriptor.effect_info());
    Some((descriptor, effect))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{CallTarget, MajitGraph, OpKind, ValueId, ValueType};

    #[test]
    fn rewrite_graph_tags_vable_fields() {
        let mut graph = MajitGraph::new("test");
        let base = graph.alloc_value();
        graph.push_op(
            graph.entry,
            OpKind::FieldRead {
                base,
                field: crate::graph::FieldDescriptor::new("next_instr", Some("Frame".into())),
                ty: ValueType::Int,
            },
            true,
        );
        graph.set_terminator(graph.entry, Terminator::Return(None));

        let config = GraphTransformConfig {
            vable_fields: vec![VirtualizableFieldDescriptor::new(
                "next_instr",
                Some("Frame".into()),
                0,
            )],
            ..Default::default()
        };
        let result = rewrite_graph(&graph, &config);
        assert_eq!(result.vable_rewrites, 1);
        // Should be rewritten to VableFieldRead
        let rewritten_op = &result.graph.block(graph.entry).ops[0];
        assert!(
            matches!(
                &rewritten_op.kind,
                OpKind::VableFieldRead { field_index: 0, .. }
            ),
            "expected VableFieldRead, got {:?}",
            rewritten_op.kind
        );
    }

    #[test]
    fn rewrite_graph_requires_matching_field_owner_root() {
        let mut graph = MajitGraph::new("test");
        let base = graph.alloc_value();
        graph.push_op(
            graph.entry,
            OpKind::FieldRead {
                base,
                field: crate::graph::FieldDescriptor::new("next_instr", Some("OtherFrame".into())),
                ty: ValueType::Int,
            },
            true,
        );
        graph.set_terminator(graph.entry, Terminator::Return(None));

        let config = GraphTransformConfig {
            vable_fields: vec![VirtualizableFieldDescriptor::new(
                "next_instr",
                Some("Frame".into()),
                0,
            )],
            ..Default::default()
        };
        let result = rewrite_graph(&graph, &config);
        assert_eq!(result.vable_rewrites, 0);
        assert!(matches!(
            result.graph.block(graph.entry).ops[0].kind,
            OpKind::FieldRead { .. }
        ));
    }

    #[test]
    fn rewrite_graph_classifies_calls() {
        let mut graph = MajitGraph::new("test");
        graph.push_op(
            graph.entry,
            OpKind::Call {
                target: CallTarget::method("call_callable", Some("PyFrame".into())),
                args: vec![],
                result_ty: ValueType::Ref,
            },
            false,
        );
        graph.set_terminator(graph.entry, Terminator::Return(None));

        let result = rewrite_graph(
            &graph,
            &crate::test_support::pyre_pipeline_config().transform,
        );
        assert_eq!(result.calls_classified, 1);
        assert!(matches!(
            result.graph.block(graph.entry).ops[0].kind,
            OpKind::CallResidual { .. }
        ));
    }

    #[test]
    fn rewrite_graph_uses_explicit_call_effect_overrides() {
        let mut graph = MajitGraph::new("test");
        graph.push_op(
            graph.entry,
            OpKind::Call {
                target: CallTarget::function_path(["custom_reader"]),
                args: vec![],
                result_ty: ValueType::Ref,
            },
            true,
        );
        graph.set_terminator(graph.entry, Terminator::Return(None));

        let result = rewrite_graph(
            &graph,
            &GraphTransformConfig {
                call_effects: vec![CallEffectOverride::new(
                    CallTarget::function_path(["custom_reader"]),
                    CallEffectKind::MayForce,
                )],
                ..Default::default()
            },
        );
        assert!(matches!(
            result.graph.block(graph.entry).ops[0].kind,
            OpKind::CallMayForce { .. }
        ));
    }

    #[test]
    fn rewrite_graph_reports_unknowns() {
        let mut graph = MajitGraph::new("demo");
        graph.push_op(
            graph.entry,
            OpKind::Unknown {
                kind: crate::graph::UnknownKind::UnsupportedExpr,
            },
            false,
        );
        graph.set_terminator(
            graph.entry,
            Terminator::Abort {
                reason: "not implemented".into(),
            },
        );
        let result = rewrite_graph(&graph, &GraphTransformConfig::default());
        assert_eq!(result.notes.len(), 2); // unknown + abort
    }

    #[test]
    fn rewrite_graph_consumes_identity_virtualizable_hints() {
        let mut graph = MajitGraph::new("demo");
        let frame = graph.alloc_value();
        let hinted = graph.alloc_value();
        graph.block_mut(graph.entry).inputargs.push(frame);
        graph.push_op(
            graph.entry,
            OpKind::Call {
                target: CallTarget::function_path(["hint_access_directly"]),
                args: vec![frame],
                result_ty: ValueType::Ref,
            },
            false,
        );
        graph.block_mut(graph.entry).ops.last_mut().unwrap().result = Some(hinted);
        graph.push_op(
            graph.entry,
            OpKind::FieldRead {
                base: hinted,
                field: crate::graph::FieldDescriptor::new("next_instr", Some("Frame".into())),
                ty: ValueType::Int,
            },
            true,
        );
        graph.set_terminator(graph.entry, Terminator::Return(None));

        let result = rewrite_graph(
            &graph,
            &GraphTransformConfig {
                vable_fields: vec![VirtualizableFieldDescriptor::new(
                    "next_instr",
                    Some("Frame".into()),
                    0,
                )],
                ..Default::default()
            },
        );

        assert_eq!(result.graph.block(graph.entry).ops.len(), 1);
        match &result.graph.block(graph.entry).ops[0].kind {
            OpKind::VableFieldRead { field_index, .. } => assert_eq!(*field_index, 0),
            other => panic!("expected VableFieldRead after hint suppression, got {other:?}"),
        }
    }

    #[test]
    fn rewrite_graph_rewrites_hint_force_virtualizable() {
        let mut graph = MajitGraph::new("demo");
        let frame = graph.alloc_value();
        let forced = graph.alloc_value();
        graph.block_mut(graph.entry).inputargs.push(frame);
        graph.push_op(
            graph.entry,
            OpKind::Call {
                target: CallTarget::function_path(["hint_force_virtualizable"]),
                args: vec![frame],
                result_ty: ValueType::Ref,
            },
            false,
        );
        graph.block_mut(graph.entry).ops.last_mut().unwrap().result = Some(forced);
        graph.push_op(
            graph.entry,
            OpKind::FieldRead {
                base: forced,
                field: crate::graph::FieldDescriptor::new("next_instr", Some("Frame".into())),
                ty: ValueType::Int,
            },
            true,
        );
        graph.set_terminator(graph.entry, Terminator::Return(None));

        let result = rewrite_graph(
            &graph,
            &GraphTransformConfig {
                vable_fields: vec![VirtualizableFieldDescriptor::new(
                    "next_instr",
                    Some("Frame".into()),
                    0,
                )],
                ..Default::default()
            },
        );

        let ops = &result.graph.block(graph.entry).ops;
        assert!(matches!(ops[0].kind, OpKind::VableForce));
        assert!(matches!(
            ops[1].kind,
            OpKind::VableFieldRead { field_index: 0, .. }
        ));
    }
}
