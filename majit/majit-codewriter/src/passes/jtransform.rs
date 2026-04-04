//! Graph-based jtransform: semantic rewrite pass.
//!
//! RPython equivalent: jtransform.py Transformer.optimize_block()
//!
//! Transforms a FunctionGraph by rewriting operations:
//! - FieldRead on virtualizable fields → VableFieldRead marker
//! - FieldWrite on virtualizable fields → VableFieldWrite marker
//! - ArrayRead on virtualizable arrays → VableArrayRead marker
//! - Call classification → elidable/residual/may_force tagging

use serde::{Deserialize, Serialize};

use crate::call::CallDescriptor;
use crate::model::{
    BlockId, CallTarget, FieldDescriptor, FunctionGraph, OpKind, SpaceOperation, Terminator,
    ValueId, ValueType,
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
    pub graph: FunctionGraph,
    pub notes: Vec<GraphTransformNote>,
    /// Number of ops rewritten by virtualizable lowering.
    pub vable_rewrites: usize,
    /// Number of calls classified.
    pub calls_classified: usize,
}

/// Rewrite a semantic graph with JIT-specific transformations.
///
/// Convenience wrapper that creates a `Transformer` and runs it.
/// RPython equivalent: jtransform.py `transform_graph()`.
pub fn rewrite_graph(graph: &FunctionGraph, config: &GraphTransformConfig) -> GraphTransformResult {
    let mut transformer = Transformer::new(config);
    transformer.transform(graph)
}

/// JIT graph transformer.
///
/// RPython equivalent: `jtransform.py` class `Transformer`.
///
/// Rewrites operations in a FunctionGraph to JIT-specific instructions:
/// - Virtualizable field/array access → VableFieldRead/VableArrayRead
/// - Hint calls → identity/VableForce
/// - Call classification → CallElidable/CallResidual/CallMayForce
pub struct Transformer<'a> {
    /// RPython: `Transformer.callcontrol`.
    callcontrol: Option<&'a mut crate::call::CallControl>,
    /// RPython: `Transformer.__init__` config for virtualizable lowering.
    config: &'a GraphTransformConfig,
    /// Type resolution state from the rtype pass.
    /// Used by `make_three_lists()` to split args by kind.
    /// RPython: types are on `Variable.concretetype` — we pass them explicitly.
    type_state: Option<&'a crate::passes::rtype::TypeResolutionState>,
    /// RPython: `Transformer.vable_array_vars`.
    vable_array_vars: std::collections::HashMap<ValueId, usize>,
    /// RPython: `Transformer.vable_flags`.
    vable_flags: std::collections::HashMap<ValueId, VableFlag>,
    /// Value aliases from identity rewrites (same_as / hint rewriting).
    aliases: std::collections::HashMap<ValueId, ValueId>,
    notes: Vec<GraphTransformNote>,
    vable_rewrites: usize,
    calls_classified: usize,
}

/// RPython: jtransform.py vable_flags values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VableFlag {
    FreshVirtualizable,
}

/// Result of rewriting a single operation.
///
/// RPython: `rewrite_operation()` returns SpaceOperation, list, None, or Constant.
enum RewriteResult {
    /// Replace with these ops
    Replace(Vec<SpaceOperation>),
    /// Remove the op (identity/alias: result remaps to the given value)
    Identity(ValueId),
    /// Keep the op unchanged
    Keep,
}

impl<'a> Transformer<'a> {
    pub fn new(config: &'a GraphTransformConfig) -> Self {
        Self {
            callcontrol: None,
            config,
            type_state: None,
            vable_array_vars: std::collections::HashMap::new(),
            vable_flags: std::collections::HashMap::new(),
            aliases: std::collections::HashMap::new(),
            notes: Vec::new(),
            vable_rewrites: 0,
            calls_classified: 0,
        }
    }

    /// Set the CallControl for call kind dispatch.
    /// RPython: `Transformer.__init__(callcontrol=...)`.
    pub fn with_callcontrol(mut self, cc: &'a mut crate::call::CallControl) -> Self {
        self.callcontrol = Some(cc);
        self
    }

    /// Set the type resolution state for arg kind splitting.
    /// RPython: types live on `Variable.concretetype`.
    pub fn with_type_state(mut self, ts: &'a crate::passes::rtype::TypeResolutionState) -> Self {
        self.type_state = Some(ts);
        self
    }

    /// RPython: Transformer.transform() — process all blocks in the graph.
    pub fn transform(&mut self, graph: &FunctionGraph) -> GraphTransformResult {
        let mut rewritten = graph.clone();

        for block in &mut rewritten.blocks {
            self.optimize_block(block, &graph.name);
        }

        GraphTransformResult {
            graph: rewritten,
            notes: std::mem::take(&mut self.notes),
            vable_rewrites: self.vable_rewrites,
            calls_classified: self.calls_classified,
        }
    }

    /// RPython: Transformer.optimize_block()
    fn optimize_block(&mut self, block: &mut crate::model::Block, graph_name: &str) {
        let mut new_ops = Vec::with_capacity(block.ops.len());

        for original_op in &block.ops {
            let op = remap_op(original_op, &self.aliases);
            match self.rewrite_operation(&op, graph_name) {
                RewriteResult::Replace(ops) => {
                    new_ops.extend(ops);
                }
                RewriteResult::Identity(alias_target) => {
                    if let Some(result) = op.result {
                        self.aliases
                            .insert(result, resolve_alias(alias_target, &self.aliases));
                    }
                }
                RewriteResult::Keep => {
                    new_ops.push(op);
                }
            }
        }

        block.ops = new_ops;
        block.terminator = remap_terminator(&block.terminator, &self.aliases);

        if let Terminator::Abort { reason } = &block.terminator {
            self.notes.push(GraphTransformNote {
                function: graph_name.to_string(),
                detail: format!("abort: {reason}"),
            });
        }
    }

    /// RPython: Transformer.rewrite_operation() — dispatch to rewrite_op_*.
    fn rewrite_operation(&mut self, op: &SpaceOperation, graph_name: &str) -> RewriteResult {
        match &op.kind {
            // ── rewrite_op_hint ──
            OpKind::Call { target, args, .. } if classify_vable_hint(target).is_some() => {
                self.rewrite_op_hint(op, target, args, graph_name)
            }
            // ── rewrite_op_getfield ──
            OpKind::FieldRead { field, ty, .. } if self.config.lower_virtualizable => {
                self.rewrite_op_getfield(op, field, ty, graph_name)
            }
            // ── rewrite_op_setfield ──
            OpKind::FieldWrite {
                field, value, ty, ..
            } if self.config.lower_virtualizable => {
                self.rewrite_op_setfield(op, field, *value, ty, graph_name)
            }
            // ── rewrite_op_getarrayitem ──
            OpKind::ArrayRead {
                base,
                index,
                item_ty,
            } if self.config.lower_virtualizable => {
                self.rewrite_op_getarrayitem(op, *base, *index, item_ty, graph_name)
            }
            // ── rewrite_op_setarrayitem ──
            OpKind::ArrayWrite {
                base,
                index,
                value,
                item_ty,
            } if self.config.lower_virtualizable => {
                self.rewrite_op_setarrayitem(op, *base, *index, *value, item_ty, graph_name)
            }
            // ── rewrite_op_direct_call ──
            OpKind::Call {
                target,
                args,
                result_ty,
            } if self.config.classify_calls => {
                self.rewrite_op_direct_call(op, target, args, result_ty, graph_name)
            }
            // ── unknown ops ──
            OpKind::Unknown { kind } => {
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!("unknown op: {:?}", kind),
                });
                RewriteResult::Keep
            }
            _ => RewriteResult::Keep,
        }
    }

    // ── helpers ──────────────────────────────────────────────

    /// RPython: `Transformer.make_three_lists(vars)` (jtransform.py:437-445).
    /// Split args into three lists by kind (int, ref, float).
    ///
    /// RPython: `add_in_correct_list(v, lst_i, lst_r, lst_f)` checks
    /// `getkind(v.concretetype)` and appends to the matching list.
    /// Void args are skipped.
    fn make_three_lists(&self, args: &[ValueId]) -> (Vec<ValueId>, Vec<ValueId>, Vec<ValueId>) {
        let mut args_i = Vec::new();
        let mut args_r = Vec::new();
        let mut args_f = Vec::new();
        for &v in args {
            let kind = self.get_value_kind(v);
            match kind {
                'i' => args_i.push(v),
                'r' => args_r.push(v),
                'f' => args_f.push(v),
                'v' => {}            // void — skip (RPython jtransform.py:449)
                _ => args_r.push(v), // unknown → ref
            }
        }
        (args_i, args_r, args_f)
    }

    /// RPython: `getkind(v.concretetype)` — get the kind of a value.
    /// Uses type_state if available, falls back to 'r' for unknown.
    fn get_value_kind(&self, v: ValueId) -> char {
        if let Some(ts) = self.type_state {
            if let Some(ct) = ts.concrete_types.get(&v) {
                return match ct {
                    crate::passes::rtype::ConcreteType::Signed => 'i',
                    crate::passes::rtype::ConcreteType::GcRef => 'r',
                    crate::passes::rtype::ConcreteType::Float => 'f',
                    crate::passes::rtype::ConcreteType::Void => 'v',
                    crate::passes::rtype::ConcreteType::Unknown => 'r',
                };
            }
        }
        'r' // default: ref (most Python values are GC refs)
    }

    // ── rewrite_op_* methods ──────────────────────────────────

    /// RPython: `Transformer.rewrite_op_hint(op)`.
    /// Dispatches based on the hint kind (access_directly, force_virtualizable,
    /// fresh_virtualizable, promote, etc.)
    fn rewrite_op_hint(
        &mut self,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[ValueId],
        graph_name: &str,
    ) -> RewriteResult {
        let hint_kind = match classify_vable_hint(target) {
            Some(k) => k,
            None => return RewriteResult::Keep,
        };
        match hint_kind {
            crate::hints::VirtualizableHintKind::AccessDirectly
            | crate::hints::VirtualizableHintKind::FreshVirtualizable => {
                // RPython: consume as identity (same_as)
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!("rewrite: {target}(...) → identity"),
                });
                if let Some(arg) = args.first().copied() {
                    RewriteResult::Identity(arg)
                } else {
                    RewriteResult::Keep
                }
            }
            crate::hints::VirtualizableHintKind::ForceVirtualizable => {
                // RPython: emit hint_force_virtualizable, preserve value as identity
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!("rewrite: {target}(...) → VableForce"),
                });
                self.vable_rewrites += 1;
                if let Some(arg) = args.first().copied() {
                    if let Some(result) = op.result {
                        self.aliases
                            .insert(result, resolve_alias(arg, &self.aliases));
                    }
                }
                RewriteResult::Replace(vec![SpaceOperation {
                    result: None,
                    kind: OpKind::VableForce,
                }])
            }
        }
    }

    /// RPython: rewrite_op_getfield
    fn rewrite_op_getfield(
        &mut self,
        op: &SpaceOperation,
        field: &FieldDescriptor,
        ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        // Track virtualizable array field reads
        if let Some(array_field) = self.config.vable_arrays.iter().find(|c| c.matches(field)) {
            if let Some(result) = op.result {
                self.vable_array_vars.insert(result, array_field.index);
            }
        }
        // Virtualizable scalar field → VableFieldRead
        if let Some(vable_field) = self.config.vable_fields.iter().find(|c| c.matches(field)) {
            self.notes.push(GraphTransformNote {
                function: graph_name.to_string(),
                detail: format!(
                    "rewrite: {} → VableFieldRead[{}]",
                    field.name, vable_field.index
                ),
            });
            self.vable_rewrites += 1;
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result,
                kind: OpKind::VableFieldRead {
                    field_index: vable_field.index,
                    ty: ty.clone(),
                },
            }]);
        }
        RewriteResult::Keep
    }

    /// RPython: rewrite_op_setfield
    fn rewrite_op_setfield(
        &mut self,
        op: &SpaceOperation,
        field: &FieldDescriptor,
        value: ValueId,
        ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        if let Some(vable_field) = self.config.vable_fields.iter().find(|c| c.matches(field)) {
            self.notes.push(GraphTransformNote {
                function: graph_name.to_string(),
                detail: format!(
                    "rewrite: {} = ... → VableFieldWrite[{}]",
                    field.name, vable_field.index
                ),
            });
            self.vable_rewrites += 1;
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result,
                kind: OpKind::VableFieldWrite {
                    field_index: vable_field.index,
                    value,
                    ty: ty.clone(),
                },
            }]);
        }
        RewriteResult::Keep
    }

    /// RPython: rewrite_op_getarrayitem
    fn rewrite_op_getarrayitem(
        &mut self,
        op: &SpaceOperation,
        base: ValueId,
        index: ValueId,
        item_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        if let Some(&arr_idx) = self.vable_array_vars.get(&base) {
            self.notes.push(GraphTransformNote {
                function: graph_name.to_string(),
                detail: format!("rewrite: array[idx] → VableArrayRead[{arr_idx}]"),
            });
            self.vable_rewrites += 1;
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result,
                kind: OpKind::VableArrayRead {
                    array_index: arr_idx,
                    elem_index: index,
                    item_ty: item_ty.clone(),
                },
            }]);
        }
        RewriteResult::Keep
    }

    /// RPython: rewrite_op_setarrayitem
    fn rewrite_op_setarrayitem(
        &mut self,
        op: &SpaceOperation,
        base: ValueId,
        index: ValueId,
        value: ValueId,
        item_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        if let Some(&arr_idx) = self.vable_array_vars.get(&base) {
            self.notes.push(GraphTransformNote {
                function: graph_name.to_string(),
                detail: format!("rewrite: array[idx] = v → VableArrayWrite[{arr_idx}]"),
            });
            self.vable_rewrites += 1;
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result,
                kind: OpKind::VableArrayWrite {
                    array_index: arr_idx,
                    elem_index: index,
                    value,
                    item_ty: item_ty.clone(),
                },
            }]);
        }
        RewriteResult::Keep
    }

    /// RPython: `Transformer.rewrite_op_direct_call(op)`.
    ///
    /// RPython jtransform.py:406-410:
    /// ```python
    /// def rewrite_op_direct_call(self, op):
    ///     kind = self.callcontrol.guess_call_kind(op)
    ///     return getattr(self, 'handle_%s_call' % kind)(op)
    /// ```
    fn rewrite_op_direct_call(
        &mut self,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        // RPython: guess_call_kind(op) → dispatch to handle_*_call
        if let Some(cc) = self.callcontrol.as_mut() {
            let kind = cc.guess_call_kind(target);
            return match kind {
                crate::call::CallKind::Regular => {
                    self.handle_regular_call(op, target, args, result_ty, graph_name)
                }
                crate::call::CallKind::Residual => {
                    // Residual: classify effect from call_effects/describe_call
                    if let Some((descriptor, effect)) =
                        classify_call(target, &self.config.call_effects)
                    {
                        match effect {
                            CallEffectKind::Elidable => self
                                .handle_elidable_call(op, descriptor, args, result_ty, graph_name),
                            CallEffectKind::MayForce => self
                                .handle_may_force_call(op, descriptor, args, result_ty, graph_name),
                            CallEffectKind::Residual => self
                                .handle_residual_call(op, descriptor, args, result_ty, graph_name),
                        }
                    } else {
                        // No effect info — plain residual
                        let descriptor = CallDescriptor::known(
                            target.clone(),
                            EffectInfo::new(ExtraEffect::CanRaise, OopSpecIndex::None),
                        );
                        self.handle_residual_call(op, descriptor, args, result_ty, graph_name)
                    }
                }
                crate::call::CallKind::Builtin => {
                    self.handle_builtin_call(op, target, args, result_ty, graph_name)
                }
                crate::call::CallKind::Recursive => {
                    self.handle_recursive_call(op, target, args, result_ty, graph_name)
                }
            };
        }

        // Fallback when no CallControl: effect-only classification (legacy path)
        if let Some((descriptor, effect)) = classify_call(target, &self.config.call_effects) {
            match effect {
                CallEffectKind::Elidable => {
                    self.handle_elidable_call(op, descriptor, args, result_ty, graph_name)
                }
                CallEffectKind::Residual => {
                    self.handle_residual_call(op, descriptor, args, result_ty, graph_name)
                }
                CallEffectKind::MayForce => {
                    self.handle_may_force_call(op, descriptor, args, result_ty, graph_name)
                }
            }
        } else {
            RewriteResult::Keep
        }
    }

    /// RPython: `Transformer.handle_builtin_call(op)`.
    /// Builtin operations with oopspec semantics — dispatched to
    /// specific lowering based on the oopspec name.
    ///
    /// RPython jtransform.py:484-520.
    ///
    /// Currently: look up effect from describe_call / call_effects
    /// and produce the matching typed call op. Future: oopspec-specific
    /// lowering (list_getitem → getarrayitem_gc, etc.)
    fn handle_builtin_call(
        &mut self,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        // Look up effect info for this builtin target.
        if let Some((descriptor, effect)) = classify_call(target, &self.config.call_effects) {
            self.notes.push(GraphTransformNote {
                function: graph_name.to_string(),
                detail: format!("builtin {target} → {}", effect.as_str()),
            });
            self.calls_classified += 1;
            match effect {
                CallEffectKind::Elidable => {
                    return RewriteResult::Replace(vec![SpaceOperation {
                        result: op.result,
                        kind: OpKind::CallElidable {
                            descriptor,
                            args: args.to_vec(),
                            result_ty: result_ty.clone(),
                        },
                    }]);
                }
                _ => {
                    return self.handle_residual_call(op, descriptor, args, result_ty, graph_name);
                }
            }
        }
        // Unknown builtin — keep as unclassified Call.
        RewriteResult::Keep
    }

    /// RPython: `Transformer.handle_regular_call(op)`.
    /// Callee is a candidate graph — emit `inline_call_*` referencing
    /// the callee's JitCode. The meta-interpreter will descend into
    /// the callee JitCode at runtime.
    ///
    /// RPython jtransform.py:473-482.
    fn handle_regular_call(
        &mut self,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        // RPython jtransform.py:477-478: get_jitcode(targetgraph)
        let jitcode_index = if let Some(cc) = self.callcontrol.as_mut() {
            let path = target_to_call_path(target);
            cc.get_jitcode(&path)
        } else {
            0
        };
        // RPython jtransform.py:480: rewrite_call(op, 'inline_call', [jitcode])
        // Split args by kind (RPython make_three_lists)
        let (args_i, args_r, args_f) = self.make_three_lists(args);
        let result_kind = value_type_to_kind(result_ty);

        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("call {target} → inline_call[jitcode={jitcode_index}]"),
        });
        self.calls_classified += 1;
        // RPython jtransform.py:480-481: inline_call always followed by -live-
        RewriteResult::Replace(vec![
            SpaceOperation {
                result: op.result,
                kind: OpKind::InlineCall {
                    jitcode_index,
                    args_i,
                    args_r,
                    args_f,
                    result_kind,
                },
            },
            SpaceOperation {
                result: None,
                kind: OpKind::Live,
            },
        ])
    }

    /// RPython: `Transformer.handle_recursive_call(op)`.
    /// Recursive call back to the portal — emit `recursive_call_*`.
    ///
    /// RPython jtransform.py:522-534.
    fn handle_recursive_call(
        &mut self,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        // RPython jtransform.py:522-534:
        //   jitdriver_sd = callcontrol.jitdriver_sd_from_portal_runner_ptr(funcptr)
        //   num_green_args = len(jitdriver_sd.jitdriver.greens)
        //   greens = args[1:1+num_green_args]
        //   reds = args[1+num_green_args:]
        //   recursive_call_{kind}(jd_index, G_I, G_R, G_F, R_I, R_R, R_F)
        let path = target_to_call_path(target);
        let (jd_index, num_green_args) = self
            .callcontrol
            .as_ref()
            .and_then(|cc| cc.jitdriver_sd_from_portal(&path))
            .map(|sd| (sd.index, sd.greens.len()))
            .unwrap_or((0, 0));

        // RPython: skip funcptr (args[0]), split rest into green/red.
        // In our AST, args don't include funcptr, so split directly.
        let green_args = if num_green_args <= args.len() {
            &args[..num_green_args]
        } else {
            args
        };
        let red_args = if num_green_args <= args.len() {
            &args[num_green_args..]
        } else {
            &[]
        };
        let (greens_i, greens_r, greens_f) = self.make_three_lists(green_args);
        let (reds_i, reds_r, reds_f) = self.make_three_lists(red_args);
        let result_kind = value_type_to_kind(result_ty);

        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!(
                "call {target} → recursive_call[jd={jd_index}, greens={num_green_args}]"
            ),
        });
        self.calls_classified += 1;

        // RPython jtransform.py:526: promote_greens emits guard_value
        // for each non-void green arg before the recursive_call.
        let mut ops = self.promote_greens(green_args);

        // RPython jtransform.py:532-533: recursive_call + -live-
        ops.push(SpaceOperation {
            result: op.result,
            kind: OpKind::RecursiveCall {
                jd_index,
                greens_i,
                greens_r,
                greens_f,
                reds_i,
                reds_r,
                reds_f,
                result_kind,
            },
        });
        ops.push(SpaceOperation {
            result: None,
            kind: OpKind::Live,
        });
        RewriteResult::Replace(ops)
    }

    /// RPython: `Transformer.promote_greens(args, jitdriver)`.
    ///
    /// Emits `-live-` + `{kind}_guard_value` for each non-void green arg.
    /// This ensures green values are constant before the recursive call.
    ///
    /// RPython jtransform.py:1646-1656.
    fn promote_greens(&self, green_args: &[ValueId]) -> Vec<SpaceOperation> {
        let mut ops = Vec::new();
        for &v in green_args {
            let kind = self.value_kind(v);
            if kind == 'v' {
                continue; // skip void
            }
            // RPython: -live- then {kind}_guard_value
            ops.push(SpaceOperation {
                result: None,
                kind: OpKind::Live,
            });
            ops.push(SpaceOperation {
                result: None,
                kind: OpKind::GuardValue {
                    value: v,
                    kind_char: kind,
                },
            });
        }
        ops
    }

    /// RPython: `getkind(v.concretetype)` — get the kind of a value.
    /// Delegates to `get_value_kind()` which consults type_state.
    fn value_kind(&self, v: ValueId) -> char {
        self.get_value_kind(v)
    }

    /// RPython: `Transformer.handle_residual_call(op)`.
    /// Call that the JIT should NOT look inside — emit residual_call_*.
    fn handle_residual_call(
        &mut self,
        op: &SpaceOperation,
        descriptor: CallDescriptor,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("call {} → residual", descriptor.target),
        });
        self.calls_classified += 1;
        // RPython jtransform.py:469-470: residual_call followed by -live-
        // if the call can raise or may call jitcodes.
        let can_raise = descriptor.effect_info.check_can_raise(false);
        let mut ops = vec![SpaceOperation {
            result: op.result,
            kind: OpKind::CallResidual {
                descriptor,
                args: args.to_vec(),
                result_ty: result_ty.clone(),
            },
        }];
        if can_raise {
            ops.push(SpaceOperation {
                result: None,
                kind: OpKind::Live,
            });
        }
        RewriteResult::Replace(ops)
    }

    /// RPython: elidable call — pure function, result depends only on args.
    fn handle_elidable_call(
        &mut self,
        op: &SpaceOperation,
        descriptor: CallDescriptor,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("call {} → elidable", descriptor.target),
        });
        self.calls_classified += 1;
        RewriteResult::Replace(vec![SpaceOperation {
            result: op.result,
            kind: OpKind::CallElidable {
                descriptor,
                args: args.to_vec(),
                result_ty: result_ty.clone(),
            },
        }])
    }

    /// RPython: may-force call — can trigger GC or force virtualizables.
    fn handle_may_force_call(
        &mut self,
        op: &SpaceOperation,
        descriptor: CallDescriptor,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("call {} → may_force", descriptor.target),
        });
        self.calls_classified += 1;
        // RPython: call_may_force always followed by -live-
        RewriteResult::Replace(vec![
            SpaceOperation {
                result: op.result,
                kind: OpKind::CallMayForce {
                    descriptor,
                    args: args.to_vec(),
                    result_ty: result_ty.clone(),
                },
            },
            SpaceOperation {
                result: None,
                kind: OpKind::Live,
            },
        ])
    }
}

/// RPython: `getkind(concretetype)[0]` → 'i', 'r', 'f', or 'v'.
///
/// RPython's rtyper resolves all types before jtransform runs, so
/// getkind() never sees an unknown type. In our pipeline, Unknown
/// means the annotate/rtype pass couldn't resolve the type. We map
/// Unknown to 'r' (ref) since most Python-level values are GC refs.
fn value_type_to_kind(ty: &ValueType) -> char {
    match ty {
        ValueType::Int | ValueType::State => 'i',
        ValueType::Ref | ValueType::Unknown => 'r',
        ValueType::Float => 'f',
        ValueType::Void => 'v',
    }
}

/// Convert a CallTarget to a CallPath for jitcode lookup.
fn target_to_call_path(target: &CallTarget) -> crate::parse::CallPath {
    match target {
        CallTarget::FunctionPath { segments } => {
            crate::parse::CallPath::from_segments(segments.iter().map(String::as_str))
        }
        CallTarget::Method { name, .. } => crate::parse::CallPath::from_segments([name.as_str()]),
        CallTarget::UnsupportedExpr => crate::parse::CallPath::from_segments(["<unsupported>"]),
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

fn remap_op(
    op: &SpaceOperation,
    aliases: &std::collections::HashMap<ValueId, ValueId>,
) -> SpaceOperation {
    let kind = match &op.kind {
        OpKind::Input { .. }
        | OpKind::ConstInt(_)
        | OpKind::VableForce
        | OpKind::Live
        | OpKind::GuardValue { .. }
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
        OpKind::InlineCall {
            jitcode_index,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => OpKind::InlineCall {
            jitcode_index: *jitcode_index,
            args_i: args_i
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            args_r: args_r
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            args_f: args_f
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            result_kind: *result_kind,
        },
        OpKind::RecursiveCall {
            jd_index,
            greens_i,
            greens_r,
            greens_f,
            reds_i,
            reds_r,
            reds_f,
            result_kind,
        } => OpKind::RecursiveCall {
            jd_index: *jd_index,
            greens_i: greens_i
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            greens_r: greens_r
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            greens_f: greens_f
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            reds_i: reds_i
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            reds_r: reds_r
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            reds_f: reds_f
                .iter()
                .copied()
                .map(|v| remap_value(v, aliases))
                .collect(),
            result_kind: *result_kind,
        },
    };
    SpaceOperation {
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
                (Some(p), Some(t)) => p == t || crate::call::is_generic_receiver(t),
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
    let descriptor = crate::call::describe_call(target)?;
    let effect = classify_effect_info(&descriptor.effect_info());
    Some((descriptor, effect))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{CallTarget, FunctionGraph, OpKind, ValueId, ValueType};

    #[test]
    fn rewrite_graph_tags_vable_fields() {
        let mut graph = FunctionGraph::new("test");
        let base = graph.alloc_value();
        graph.push_op(
            graph.entry,
            OpKind::FieldRead {
                base,
                field: crate::model::FieldDescriptor::new("next_instr", Some("Frame".into())),
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
        let mut graph = FunctionGraph::new("test");
        let base = graph.alloc_value();
        graph.push_op(
            graph.entry,
            OpKind::FieldRead {
                base,
                field: crate::model::FieldDescriptor::new("next_instr", Some("OtherFrame".into())),
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
        let mut graph = FunctionGraph::new("test");
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
        let mut graph = FunctionGraph::new("test");
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
        let mut graph = FunctionGraph::new("demo");
        graph.push_op(
            graph.entry,
            OpKind::Unknown {
                kind: crate::model::UnknownKind::UnsupportedExpr,
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
        let mut graph = FunctionGraph::new("demo");
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
                field: crate::model::FieldDescriptor::new("next_instr", Some("Frame".into())),
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
        let mut graph = FunctionGraph::new("demo");
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
                field: crate::model::FieldDescriptor::new("next_instr", Some("Frame".into())),
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
