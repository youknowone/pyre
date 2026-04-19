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
    CallTarget, FieldDescriptor, FunctionGraph, OpKind, SpaceOperation, Terminator, ValueId,
    ValueType,
};
use majit_ir::descr::{EffectInfo, ExtraEffect, OopSpecIndex};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VirtualizableFieldDescriptor {
    pub name: String,
    pub owner_root: Option<String>,
    pub index: usize,
    /// RPython: cpu.arraydescrof(ARRAY.TO).itemsize — item byte size for
    /// vable arrays. None for scalar fields.
    pub array_itemsize: Option<usize>,
    /// RPython: arraydescr.is_item_signed() — FLAG_SIGNED for vable arrays.
    pub array_is_signed: Option<bool>,
}

impl VirtualizableFieldDescriptor {
    pub fn new(name: impl Into<String>, owner_root: Option<String>, index: usize) -> Self {
        Self {
            name: name.into(),
            owner_root,
            index,
            array_itemsize: None,
            array_is_signed: None,
        }
    }

    /// Create a descriptor with arraydescr info (for vable array fields).
    /// RPython: `VirtualizableInfo.__init__` stores `cpu.arraydescrof(ARRAY.TO)`.
    pub fn new_with_arraydescr(
        name: impl Into<String>,
        owner_root: Option<String>,
        index: usize,
        itemsize: usize,
        is_signed: bool,
    ) -> Self {
        Self {
            name: name.into(),
            owner_root,
            index,
            array_itemsize: Some(itemsize),
            array_is_signed: Some(is_signed),
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
    /// `op.args[0]`-equivalent funcptr identity used to match the
    /// override against a call site.
    pub target: CallTarget,
    /// `calldescr`-equivalent EffectInfo wrapper attached to the call.
    pub descriptor: CallDescriptor,
}

impl CallEffectOverride {
    pub fn new(target: CallTarget, effect: CallEffectKind) -> Self {
        Self {
            target,
            descriptor: CallDescriptor::override_effect(effect_info_for_kind(effect)),
        }
    }
}

fn effect_info_for_kind(effect: CallEffectKind) -> EffectInfo {
    match effect {
        CallEffectKind::Elidable => {
            EffectInfo::new(ExtraEffect::ElidableCannotRaise, OopSpecIndex::None)
        }
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
    type_state: Option<&'a crate::translate_legacy::rtyper::rtyper::TypeResolutionState>,
    /// RPython: `Transformer.vable_array_vars`.
    /// Stores (array_index, itemsize, is_signed) per vable array variable.
    vable_array_vars: std::collections::HashMap<ValueId, (usize, usize, bool)>,
    /// RPython: `Transformer.vable_flags`.
    #[allow(dead_code)]
    vable_flags: std::collections::HashMap<ValueId, VableFlag>,
    /// Value aliases from identity rewrites (same_as / hint rewriting).
    aliases: std::collections::HashMap<ValueId, ValueId>,
    notes: Vec<GraphTransformNote>,
    vable_rewrites: usize,
    calls_classified: usize,
    /// RPython: DependencyTracker — caches transitive analysis results.
    /// Shared across all getcalldescr() calls within this transform pass.
    analysis_cache: crate::call::AnalysisCache,
    /// Cursor into the running graph's `next_value` counter.  Kept in
    /// sync with `FunctionGraph::next_value`: seeded from
    /// `graph.next_value` at `transform()` entry and written back at
    /// exit.  `rewrite_op_indirect_call` uses this to synthesize a
    /// fresh ValueId for the `FuncptrFromVtable` result (RPython's
    /// implicit `Variable()` allocation in `jtransform.py:546`).
    next_synthetic_value: usize,
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
            analysis_cache: crate::call::AnalysisCache::default(),
            next_synthetic_value: 0,
        }
    }

    /// Allocate a fresh ValueId synchronized with the graph's
    /// `next_value` counter.  Used by rewrite rules that need to
    /// introduce a new intermediate value (e.g. `FuncptrFromVtable`).
    fn allocate_synthetic_value(&mut self) -> ValueId {
        let id = ValueId(self.next_synthetic_value);
        self.next_synthetic_value += 1;
        id
    }

    /// Set the CallControl for call kind dispatch.
    /// RPython: `Transformer.__init__(callcontrol=...)`.
    pub fn with_callcontrol(mut self, cc: &'a mut crate::call::CallControl) -> Self {
        self.callcontrol = Some(cc);
        self
    }

    /// Set the type resolution state for arg kind splitting.
    /// RPython: types live on `Variable.concretetype`.
    pub fn with_type_state(
        mut self,
        ts: &'a crate::translate_legacy::rtyper::rtyper::TypeResolutionState,
    ) -> Self {
        self.type_state = Some(ts);
        self
    }

    /// RPython: Transformer.transform() — process all blocks in the graph.
    pub fn transform(&mut self, graph: &FunctionGraph) -> GraphTransformResult {
        let mut rewritten = graph.clone();
        // Keep the ValueId allocator in sync with the graph we're
        // rewriting so rules that synthesize values stay unique.
        self.next_synthetic_value = rewritten.next_value();

        for block in &mut rewritten.blocks {
            self.optimize_block(block, &graph.name);
        }

        rewritten.set_next_value(self.next_synthetic_value);

        GraphTransformResult {
            graph: rewritten,
            notes: std::mem::take(&mut self.notes),
            vable_rewrites: self.vable_rewrites,
            calls_classified: self.calls_classified,
        }
    }

    /// RPython: Transformer.optimize_block()
    fn optimize_block(&mut self, block: &mut crate::model::Block, graph_name: &str) {
        let mut new_ops = Vec::with_capacity(block.operations.len());

        for original_op in &block.operations {
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

        block.operations = new_ops;
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
            //
            // Unlike the setfield/getarrayitem dispatch this runs whether or
            // not `lower_virtualizable` is enabled: the quasi-immutable
            // `-live-` + `record_quasiimmut_field` pair from
            // `rpython/jit/codewriter/jtransform.py:895-903` is independent
            // of virtualizable lowering.  `rewrite_op_getfield` internally
            // falls through to `RewriteResult::Keep` for mutable fields and
            // plain immutables (their purity is carried on the descriptor).
            OpKind::FieldRead { field, ty, .. } => {
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
                ..
            } if self.config.lower_virtualizable => {
                self.rewrite_op_getarrayitem(op, *base, *index, item_ty, graph_name)
            }
            // ── rewrite_op_setarrayitem ──
            OpKind::ArrayWrite {
                base,
                index,
                value,
                item_ty,
                ..
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
                    crate::translate_legacy::rtyper::rtyper::ConcreteType::Signed => 'i',
                    crate::translate_legacy::rtyper::rtyper::ConcreteType::GcRef => 'r',
                    crate::translate_legacy::rtyper::rtyper::ConcreteType::Float => 'f',
                    crate::translate_legacy::rtyper::rtyper::ConcreteType::Void => 'v',
                    crate::translate_legacy::rtyper::rtyper::ConcreteType::Unknown => 'r',
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

    /// RPython `rpython/jit/codewriter/jtransform.py:830-906 rewrite_op_getfield`.
    ///
    /// Virtualizable lowering takes precedence (RPython `self.vable_array_vars`
    /// tracking + immediate return).  Otherwise the field's immutability rank
    /// drives the emit shape:
    ///
    /// * `IR_IMMUTABLE`           → rewrite the read to
    ///   `getfield_*_pure` (`jtransform.py:875-877`).
    /// * `IR_QUASIIMMUTABLE[_ARRAY]` → emit `[-live-, record_quasiimmut_field,
    ///   getfield_*_pure]` — `jtransform.py:895-903`.
    /// * mutable                  → keep as-is.
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
                // RPython: vable_array_vars[result] = (v_base, arrayfielddescr, arraydescr)
                // We store (index, itemsize, is_signed) — the arraydescr properties.
                let itemsize = array_field.array_itemsize.unwrap_or(8);
                let is_signed = array_field.array_is_signed.unwrap_or(false);
                self.vable_array_vars
                    .insert(result, (array_field.index, itemsize, is_signed));
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
        // `jtransform.py:867-903` — immutable and quasi-immutable
        // field reads both become `getfield_*_pure`; the quasi variant
        // additionally prepends `-live-` + `record_quasiimmut_field`.
        //    return [SpaceOperation('-live-', [], None),
        //            SpaceOperation('record_quasiimmut_field',
        //                           [v_inst, descr, descr1], None),
        //            op1]       # op1 = getfield_*_pure
        // Mutable fields stay as plain `getfield_gc_*`.
        let rank = self
            .callcontrol
            .as_deref()
            .and_then(|cc| cc.field_immutability(field.owner_root.as_deref(), &field.name));
        if let Some(rank) = rank {
            let OpKind::FieldRead {
                base,
                field: _,
                ty: _,
                pure: _,
            } = &op.kind
            else {
                return RewriteResult::Keep;
            };
            if rank.is_quasi_immutable() {
                // PRE-EXISTING-ADAPTATION: RPython
                // `quasiimmut.get_mutate_field_name(fieldname)` —
                // `rpython/jit/metainterp/quasiimmut.py:11-15` — strips the
                // lltype `inst_` prefix before prepending `mutate_`.  Rust
                // structs carry no such prefix, so we prepend `mutate_`
                // directly.
                let mutate_field = FieldDescriptor::new(
                    format!("mutate_{}", field.name),
                    field.owner_root.clone(),
                );
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!(
                        "rewrite: getfield({owner}.{name}) → -live- + record_quasiimmut_field + pure read",
                        owner = field.owner_root.as_deref().unwrap_or("<?>"),
                        name = field.name,
                    ),
                });
                return RewriteResult::Replace(vec![
                    SpaceOperation {
                        result: None,
                        kind: OpKind::Live,
                    },
                    SpaceOperation {
                        result: None,
                        kind: OpKind::RecordQuasiImmutField {
                            base: *base,
                            field: field.clone(),
                            mutate_field,
                        },
                    },
                    SpaceOperation {
                        result: op.result,
                        kind: OpKind::FieldRead {
                            base: *base,
                            field: field.clone(),
                            ty: ty.clone(),
                            pure: true,
                        },
                    },
                ]);
            }
            if rank.is_immutable() {
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!(
                        "rewrite: getfield({owner}.{name}) → pure read",
                        owner = field.owner_root.as_deref().unwrap_or("<?>"),
                        name = field.name,
                    ),
                });
                return RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result,
                    kind: OpKind::FieldRead {
                        base: *base,
                        field: field.clone(),
                        ty: ty.clone(),
                        pure: true,
                    },
                }]);
            }
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
        if let Some(&(arr_idx, itemsize, is_signed)) = self.vable_array_vars.get(&base) {
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
                    array_itemsize: itemsize,
                    array_is_signed: is_signed,
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
        if let Some(&(arr_idx, itemsize, is_signed)) = self.vable_array_vars.get(&base) {
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
                    array_itemsize: itemsize,
                    array_is_signed: is_signed,
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
        // RPython `jtransform.py:406-412`:
        //   def rewrite_op_direct_call(op):  ... handle_%s_call
        //   def rewrite_op_indirect_call(op): ... handle_%s_indirect_call
        //
        // Our front-end distinguishes the two via `CallTarget::Indirect`
        // rather than opname; re-enter the indirect entry point here so
        // the regular vs residual split follows `guess_call_kind`.
        if matches!(target, CallTarget::Indirect { .. }) {
            return self.rewrite_op_indirect_call(op, target, args, result_ty, graph_name);
        }
        // RPython: guess_call_kind(op) → dispatch to handle_*_call
        if let Some(cc) = self.callcontrol.as_mut() {
            let kind = cc.guess_call_kind(target);
            return match kind {
                crate::call::CallKind::Regular => {
                    self.handle_regular_call(op, target, args, result_ty, graph_name)
                }
                crate::call::CallKind::Residual => {
                    // RPython jtransform.py:456-471:
                    //   calldescr = self.callcontrol.getcalldescr(op, ...)
                    //   op1 = self.rewrite_call(op, 'residual_call', ...)
                    //
                    // RPython ALWAYS produces residual_call_* for residual
                    // calls — the effect is only in the calldescr, NOT in
                    // the opcode name. No dispatch_by_effect.
                    let descriptor =
                        if let Some((d, _)) = classify_call(target, &self.config.call_effects) {
                            d
                        } else {
                            // RPython call.py:220-222: NON_VOID_ARGS + RESULT
                            let non_void_args = resolve_non_void_arg_types(args, self.type_state);
                            let result_ir_type = value_type_to_ir_type(result_ty);
                            let cc_ref: &crate::call::CallControl =
                                self.callcontrol.as_deref().unwrap();
                            cc_ref.getcalldescr(
                                target,
                                non_void_args,
                                result_ir_type,
                                OopSpecIndex::None,
                                None,
                                &mut self.analysis_cache,
                            )
                        };
                    self.handle_residual_call(op, target, descriptor, args, result_ty, graph_name)
                }
                crate::call::CallKind::Builtin => {
                    self.handle_builtin_call(op, target, args, result_ty, graph_name)
                }
                crate::call::CallKind::Recursive => {
                    self.handle_recursive_call(op, target, args, result_ty, graph_name)
                }
            };
        }

        // Fallback when no CallControl: effect-only classification (legacy path).
        // RPython: always residual_call_*, effect only in calldescr.
        if let Some((descriptor, _effect)) = classify_call(target, &self.config.call_effects) {
            self.handle_residual_call(op, target, descriptor, args, result_ty, graph_name)
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
        // RPython jtransform.py:484-520: __handle_oopspec_call pattern.
        //   calldescr = self.callcontrol.getcalldescr(op, oopspecindex, extraeffect)
        //   self.callcontrol.callinfocollection.add(oopspecindex, calldescr, func)

        // Look up oopspec/effect info: config overrides first, then static table.
        // rlib/jit.py:250 — user-level `@oopspec(spec)` is registered via
        // `call_control.mark_oopspec(path, spec)`; look it up here before
        // falling back to the static builtin table.
        let user_oopspec: Option<String> = self
            .callcontrol
            .as_deref()
            .and_then(|cc| cc.get_oopspec(target).map(|s| s.to_string()));

        // jtransform.py:497-498 — oopspec dispatch by prefix.
        if let Some(spec) = user_oopspec.as_deref() {
            let base = spec.split('(').next().unwrap_or(spec).trim();
            // jtransform.py:497 — jit.* oopspecs → __handle_jit_call
            if base.starts_with("jit.") {
                return self._handle_jit_call(base, op, target, args, result_ty, graph_name);
            }
            // NOTE: conditional_call!/conditional_call_elidable!/record_known_result!
            // are handled by jitcode_lower (proc-macro level), NOT here.
            // The codewriter AST parser does not expand macro_rules!, so these
            // macros appear as Stmt::Macro → UnknownKind::MacroStmt.
            // The jitcode_lower proc-macro intercepts the macros directly and
            // emits BC_COND_CALL_* / BC_RECORD_KNOWN_RESULT_* bytecodes.
        }
        let (oopspecindex, extraeffect_override) =
            if let Some((descriptor, _)) = classify_call(target, &self.config.call_effects) {
                (
                    descriptor.extra_info.oopspecindex,
                    Some(descriptor.extra_info.extraeffect),
                )
            } else if let Some(descriptor) = crate::call::describe_call(target) {
                (
                    descriptor.extra_info.oopspecindex,
                    Some(descriptor.extra_info.extraeffect),
                )
            } else if let Some(spec) = user_oopspec.as_deref() {
                // rlib/jit.py:250 — map user oopspec string to OopSpecIndex.
                // jtransform.py:1731-1755 — jit.* oopspecs.
                let idx = map_user_oopspec_to_index(spec);
                (idx, None)
            } else {
                // Unknown builtin — keep as unclassified Call.
                return RewriteResult::Keep;
            };

        // RPython jtransform.py:1990-2002:
        //   calldescr = self.callcontrol.getcalldescr(op, oopspecindex, extraeffect)
        //
        // RPython reuses the same calldescr for both the op and callinfocollection.
        // We compute arg types once and clone for the collection.
        let non_void_args = resolve_non_void_arg_types(args, self.type_state);
        let result_ir_type = value_type_to_ir_type(result_ty);
        let non_void_args_for_collection = non_void_args.clone();
        let descriptor = {
            let cc_ref: &crate::call::CallControl = self.callcontrol.as_deref().unwrap();
            cc_ref.getcalldescr(
                target,
                non_void_args,
                result_ir_type,
                oopspecindex,
                extraeffect_override,
                &mut self.analysis_cache,
            )
        };

        let effect_str = format!("{:?}", descriptor.extra_info.extraeffect);
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("builtin {target} → {effect_str}"),
        });
        self.calls_classified += 1;

        // RPython jtransform.py:2000-2002:
        //   func = ptr2int(op.args[0].value)
        //   self.callcontrol.callinfocollection.add(oopspecindex, calldescr, func)
        //
        // RPython reuses the SAME calldescr returned by getcalldescr() —
        // it carries the real NON_VOID_ARGS and RESULT types from call.py:334.
        if oopspecindex != OopSpecIndex::None {
            if let Some(cc) = self.callcontrol.as_mut() {
                let calldescr: majit_ir::descr::DescrRef = majit_ir::descr::make_call_descr(
                    non_void_args_for_collection,
                    result_ir_type,
                    descriptor.extra_info.clone(),
                );

                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                target.hash(&mut hasher);
                let func_as_int = hasher.finish();

                cc.callinfocollection
                    .add(oopspecindex, calldescr, func_as_int);
                cc.callinfocollection
                    .register_func_name(func_as_int, format!("{target}"));
            }
        }

        // RPython jtransform.py:2003-2007: __handle_oopspec_call always
        // produces residual_call_*, appends -live- if calldescr_canraise.
        // Effect is only in the calldescr, never in the opcode name.
        self.handle_residual_call(op, target, descriptor, args, result_ty, graph_name)
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
            .and_then(|cc| cc.jitdriver_sd_from_portal_graph(&path))
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

    /// RPython: `Transformer.__handle_jit_call(op, oopspec_name, args)` (jtransform.py:1730-1757).
    /// Dispatches jit.* oopspec calls to dedicated opcodes or __handle_oopspec_call.
    fn _handle_jit_call(
        &mut self,
        oopspec_name: &str,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        match oopspec_name {
            // jtransform.py:1731-1732
            "jit.debug" => {
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: "jit.debug → jit_debug".to_string(),
                });
                RewriteResult::Replace(vec![SpaceOperation {
                    result: None,
                    kind: OpKind::JitDebug {
                        args: args.to_vec(),
                    },
                }])
            }
            // jtransform.py:1733-1735
            "jit.assert_green" => {
                let value = args[0];
                let kind_char = self.get_value_kind(value);
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!("jit.assert_green → {kind_char}_assert_green"),
                });
                RewriteResult::Replace(vec![SpaceOperation {
                    result: None,
                    kind: OpKind::AssertGreen { value, kind_char },
                }])
            }
            // jtransform.py:1736-1737
            "jit.current_trace_length" => {
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: "jit.current_trace_length → current_trace_length".to_string(),
                });
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result,
                    kind: OpKind::CurrentTraceLength,
                }])
            }
            // jtransform.py:1738-1740
            "jit.isconstant" => {
                let value = args[0];
                let kind_char = self.get_value_kind(value);
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!("jit.isconstant → {kind_char}_isconstant"),
                });
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result,
                    kind: OpKind::IsConstant { value, kind_char },
                }])
            }
            // jtransform.py:1741-1743
            "jit.isvirtual" => {
                let value = args[0];
                let kind_char = self.get_value_kind(value);
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!("jit.isvirtual → {kind_char}_isvirtual"),
                });
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result,
                    kind: OpKind::IsVirtual { value, kind_char },
                }])
            }
            // jtransform.py:1744-1747
            "jit.force_virtual" => self._handle_oopspec_call(
                op,
                target,
                args,
                result_ty,
                graph_name,
                OopSpecIndex::JitForceVirtual,
                Some(majit_ir::descr::ExtraEffect::ForcesVirtualOrVirtualizable),
            ),
            // jtransform.py:1748-1755
            "jit.not_in_trace" => {
                // jtransform.py:1750-1753: not_in_trace must return void
                assert!(
                    *result_ty == ValueType::Void,
                    "jit.not_in_trace() function must return None"
                );
                self._handle_oopspec_call(
                    op,
                    target,
                    args,
                    result_ty,
                    graph_name,
                    OopSpecIndex::NotInTrace,
                    None,
                )
            }
            // jtransform.py:1756-1757
            _ => {
                // jtransform.py:1757
                panic!("missing support for jit.* oopspec: {oopspec_name}");
            }
        }
    }

    /// RPython: `Transformer.__handle_oopspec_call(op, args, oopspecindex, extraeffect)`
    /// (jtransform.py:1988-2008).
    /// Produces a residual_call with the given oopspecindex embedded in the calldescr,
    /// and registers the function in the callinfocollection.
    fn _handle_oopspec_call(
        &mut self,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
        oopspecindex: OopSpecIndex,
        extraeffect: Option<majit_ir::descr::ExtraEffect>,
    ) -> RewriteResult {
        // jtransform.py:1990-1993
        let non_void_args = resolve_non_void_arg_types(args, self.type_state);
        let result_ir_type = value_type_to_ir_type(result_ty);
        let non_void_args_for_collection = non_void_args.clone();
        let descriptor = {
            let cc_ref: &crate::call::CallControl = self.callcontrol.as_deref().unwrap();
            cc_ref.getcalldescr(
                target,
                non_void_args,
                result_ir_type,
                oopspecindex,
                extraeffect,
                &mut self.analysis_cache,
            )
        };
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("oopspec {oopspecindex:?} → residual_call"),
        });
        self.calls_classified += 1;
        // jtransform.py:1999-2002: callinfocollection.add
        if oopspecindex != OopSpecIndex::None {
            if let Some(cc) = self.callcontrol.as_mut() {
                let calldescr = majit_ir::descr::make_call_descr(
                    non_void_args_for_collection,
                    result_ir_type,
                    descriptor.extra_info.clone(),
                );
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                target.hash(&mut hasher);
                let func_as_int = hasher.finish();
                cc.callinfocollection
                    .add(oopspecindex, calldescr, func_as_int);
                cc.callinfocollection
                    .register_func_name(func_as_int, format!("{target}"));
            }
        }
        // jtransform.py:2003-2008: residual_call + optional -live-
        self.handle_residual_call(op, target, descriptor, args, result_ty, graph_name)
    }

    // NOTE: rewrite_op_jit_conditional_call, _rewrite_op_cond_call, and
    // rewrite_op_jit_record_known_result are handled by jitcode_lower
    // (proc-macro level), not jtransform. The codewriter AST parser does
    // not expand macro_rules!, so these macros never reach jtransform.
    // See jitcode_lower.rs: lower_conditional_call, lower_conditional_call_elidable,
    // lower_record_known_result.
    //
    // `_rewrite_op_cond_call` below is a structural mirror of
    // `rpython/jit/codewriter/jtransform.py:1665-1683`. pyre dispatches
    // conditional_call via the proc-macro path (see above), so this
    // function is never reached at runtime; the Rust #[allow(dead_code)]
    // is a deliberate PRE-EXISTING-ADAPTATION marker. Keeping the body
    // here lets future porters cross-reference our conditional_call
    // lowering against the upstream flow line-by-line.

    /// RPython: `Transformer._rewrite_op_cond_call(op, rewritten_opname)`
    /// (jtransform.py:1665-1683).
    ///
    /// Called by upstream `rewrite_op_jit_conditional_call` and
    /// `rewrite_op_jit_conditional_call_value`; in pyre those two
    /// lower through `jitcode_lower::lower_conditional_call` /
    /// `lower_conditional_call_elidable` instead. This body is kept as
    /// structural documentation so the two code paths stay aligned.
    #[allow(dead_code)]
    fn _rewrite_op_cond_call(
        &mut self,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
        is_value: bool,
    ) -> RewriteResult {
        // jtransform.py:1666-1672: validate no floats, ≤4+2 args
        for &arg in args {
            if self.get_value_kind(arg) == 'f' {
                panic!("Conditional call does not support floats");
            }
        }
        if args.len() > 4 + 2 {
            panic!("Conditional call does not support more than 4 arguments");
        }
        // jtransform.py:1673-1676: calldescr from function call (args[1:] → result)
        let condition_or_value = args[0];
        let func_args = if args.len() > 1 { &args[1..] } else { &[] };
        let non_void_args = resolve_non_void_arg_types(func_args, self.type_state);
        let result_ir_type = value_type_to_ir_type(result_ty);
        let descriptor = {
            let cc_ref: &crate::call::CallControl = self.callcontrol.as_deref().unwrap();
            cc_ref.getcalldescr(
                target,
                non_void_args,
                result_ir_type,
                OopSpecIndex::None,
                None,
                &mut self.analysis_cache,
            )
        };
        // jtransform.py:1677: assert not forces_virtual_or_virtualizable
        assert!(
            !descriptor
                .extra_info
                .check_forces_virtual_or_virtualizable(),
            "conditional_call target must not force virtualizable"
        );
        // jtransform.py:1678-1680: rewrite_call with force_ir=True
        let (args_i, args_r, args_f) = self.make_three_lists(func_args);
        assert!(
            args_f.is_empty(),
            "force_ir: no float args in conditional_call"
        );
        let result_kind = value_type_to_kind(result_ty);
        let rewritten_opname = if is_value {
            "conditional_call_value"
        } else {
            "conditional_call"
        };
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("{rewritten_opname} → {rewritten_opname}_ir_{result_kind}"),
        });
        self.calls_classified += 1;
        let call_kind = if is_value {
            OpKind::ConditionalCallValue {
                value: condition_or_value,
                funcptr: target.clone(),
                descriptor: descriptor.clone(),
                args_i,
                args_r,
                args_f,
                result_kind,
            }
        } else {
            OpKind::ConditionalCall {
                condition: condition_or_value,
                funcptr: target.clone(),
                descriptor: descriptor.clone(),
                args_i,
                args_r,
                args_f,
            }
        };
        // jtransform.py:1681-1682: -live- if calldescr_canraise
        let mut ops = vec![SpaceOperation {
            result: op.result,
            kind: call_kind,
        }];
        if descriptor.extra_info.check_can_raise(false) {
            ops.push(SpaceOperation {
                result: None,
                kind: OpKind::Live,
            });
        }
        RewriteResult::Replace(ops)
    }

    /// RPython: `Transformer.rewrite_op_jit_conditional_call(op)`
    /// (jtransform.py:1685-1686). Dispatch wrapper kept for structural
    /// parity; pyre's `rewrite_operation` match does not reach it.
    #[allow(dead_code)]
    fn rewrite_op_jit_conditional_call(
        &mut self,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        self._rewrite_op_cond_call(op, target, args, result_ty, graph_name, false)
    }

    /// RPython: `Transformer.rewrite_op_jit_conditional_call_value(op)`
    /// (jtransform.py:1687-1688). Dispatch wrapper kept for structural
    /// parity; pyre's `rewrite_operation` match does not reach it.
    #[allow(dead_code)]
    fn rewrite_op_jit_conditional_call_value(
        &mut self,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        self._rewrite_op_cond_call(op, target, args, result_ty, graph_name, true)
    }

    /// RPython: `Transformer.rewrite_op_jit_record_known_result(op)`
    /// (jtransform.py:292-313).
    #[allow(dead_code)]
    fn rewrite_op_jit_record_known_result(
        &mut self,
        _op: &SpaceOperation,
        target: &CallTarget,
        args: &[ValueId],
        _result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        // jtransform.py:293-295: validate no floats
        for &arg in args {
            if self.get_value_kind(arg) == 'f' {
                panic!("record_known_result does not support floats");
            }
        }
        // jtransform.py:298-300: calldescr from function (args[1:] → args[0])
        // args[0] = known result, args[1..] = function args
        let result_value = args[0];
        let func_args = if args.len() > 1 { &args[1..] } else { &[] };
        let result_kind = self.get_value_kind(result_value);
        let result_ir_type = match result_kind {
            'i' => majit_ir::value::Type::Int,
            'r' => majit_ir::value::Type::Ref,
            _ => {
                panic!("record_known_result: unsupported result kind '{result_kind}'");
            }
        };
        let non_void_args = resolve_non_void_arg_types(func_args, self.type_state);
        let descriptor = {
            let cc_ref: &crate::call::CallControl = self.callcontrol.as_deref().unwrap();
            cc_ref.getcalldescr(
                target,
                non_void_args,
                result_ir_type,
                OopSpecIndex::None,
                None,
                &mut self.analysis_cache,
            )
        };
        // jtransform.py:301: assert calldescr.get_extra_info().check_is_elidable()
        assert!(
            descriptor.extra_info.check_is_elidable(),
            "record_known_result: function must be elidable"
        );
        // jtransform.py:302-307: record_known_result_{i|r}
        let opname = format!("record_known_result_{result_kind}");
        // jtransform.py:308-310: rewrite_call with force_ir=True
        let (args_i, args_r, args_f) = self.make_three_lists(func_args);
        assert!(
            args_f.is_empty(),
            "force_ir: no float args in record_known_result"
        );
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("{opname} → {opname}_ir_v"),
        });
        self.calls_classified += 1;
        // jtransform.py:311-313: -live- if calldescr_canraise
        let mut ops = vec![SpaceOperation {
            result: None, // record_known_result produces void
            kind: OpKind::RecordKnownResult {
                result_value,
                funcptr: target.clone(),
                descriptor: descriptor.clone(),
                args_i,
                args_r,
                args_f,
                result_kind,
            },
        }];
        if descriptor.extra_info.check_can_raise(false) {
            ops.push(SpaceOperation {
                result: None,
                kind: OpKind::Live,
            });
        }
        RewriteResult::Replace(ops)
    }

    /// RPython: `Transformer.handle_residual_call(op)` (jtransform.py:456-471).
    /// Call that the JIT should NOT look inside — emit residual_call_*.
    /// Args are split by kind via `rewrite_call()` → `make_three_lists()`.
    /// `target` is the funcptr identity (mirrors `op.args[0]` upstream),
    /// kept separate from `descriptor` per jtransform.py:457.
    fn handle_residual_call(
        &mut self,
        op: &SpaceOperation,
        target: &CallTarget,
        descriptor: CallDescriptor,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        self.handle_residual_call_with_targets(
            op, target, descriptor, args, result_ty, graph_name, None,
        )
    }

    /// RPython `jtransform.py:456-471` + `jtransform.py:547` sidecar:
    /// the `IndirectCallTargets(lst)` passed via `extraargs` rides along
    /// with the residual_call opcode.  This variant exposes the
    /// `indirect_targets` parameter so `handle_regular_indirect_call`
    /// can attach the candidate jitcode list without having to build the
    /// `OpKind::CallResidual` twice.
    fn handle_residual_call_with_targets(
        &mut self,
        op: &SpaceOperation,
        target: &CallTarget,
        descriptor: CallDescriptor,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
        indirect_targets: Option<crate::model::IndirectCallTargets>,
    ) -> RewriteResult {
        let note_detail = match &indirect_targets {
            Some(t) => format!(
                "call {target} → residual indirect ({} candidates)",
                t.candidates.len()
            ),
            None => format!("call {target} → residual"),
        };
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: note_detail,
        });
        self.calls_classified += 1;
        // RPython jtransform.py:467: rewrite_call(op, 'residual_call', ...)
        let (args_i, args_r, args_f) = self.make_three_lists(args);
        let result_kind = value_type_to_kind(result_ty);
        // RPython jtransform.py:469-470: residual_call followed by -live-
        // if the call can raise or may call jitcodes.
        // jtransform.py:547: `handle_regular_indirect_call` passes
        // `may_call_jitcodes=True`, which forces a trailing `-live-`.
        let can_raise = descriptor.extra_info.check_can_raise(false) || indirect_targets.is_some();
        let mut ops = vec![SpaceOperation {
            result: op.result,
            kind: OpKind::CallResidual {
                funcptr: target.clone(),
                descriptor,
                args_i,
                args_r,
                args_f,
                result_kind,
                indirect_targets,
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

    /// RPython `jtransform.py:410-412`:
    /// ```python
    /// def rewrite_op_indirect_call(self, op):
    ///     kind = self.callcontrol.guess_call_kind(op)
    ///     return getattr(self, 'handle_%s_indirect_call' % kind)(op)
    /// ```
    ///
    /// Regular  → `handle_regular_indirect_call` (guard + residual).
    /// Residual → `handle_residual_indirect_call` (alias for
    ///            `handle_residual_call`, `jtransform.py:536`).
    ///
    /// Builtin/Recursive branches don't arise for `dyn Trait` dispatch
    /// in the Rust front-end today; if we see one, fall back to the
    /// residual path to stay safe.
    fn rewrite_op_indirect_call(
        &mut self,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        let Some(cc) = self.callcontrol.as_deref() else {
            // No CallControl → leave the op untouched (residual handling
            // kicks in downstream if the effect table has an entry).
            if let Some((descriptor, _)) = classify_call(target, &self.config.call_effects) {
                return self
                    .handle_residual_call(op, target, descriptor, args, result_ty, graph_name);
            }
            return RewriteResult::Keep;
        };
        let kind = cc.guess_call_kind(target);
        match kind {
            crate::call::CallKind::Regular => {
                self.handle_regular_indirect_call(op, target, args, result_ty, graph_name)
            }
            // jtransform.py:536 `handle_residual_indirect_call = handle_residual_call`.
            crate::call::CallKind::Residual
            | crate::call::CallKind::Builtin
            | crate::call::CallKind::Recursive => {
                let non_void_args = resolve_non_void_arg_types(args, self.type_state);
                let result_ir_type = value_type_to_ir_type(result_ty);
                let cc_ref: &crate::call::CallControl = self.callcontrol.as_deref().unwrap();
                let descriptor = cc_ref.getcalldescr(
                    target,
                    non_void_args,
                    result_ir_type,
                    OopSpecIndex::None,
                    None,
                    &mut self.analysis_cache,
                );
                self.handle_residual_call(op, target, descriptor, args, result_ty, graph_name)
            }
        }
    }

    /// RPython `jtransform.py:538-553`:
    /// ```python
    /// def handle_regular_indirect_call(self, op):
    ///     lst = []
    ///     for targetgraph in self.callcontrol.graphs_from(op):
    ///         lst.append(self.callcontrol.get_jitcode(targetgraph, ...))
    ///     op0 = SpaceOperation('-live-', [], None)
    ///     op1 = SpaceOperation('int_guard_value', [op.args[0]], None)
    ///     op2 = self.handle_residual_call(op, [IndirectCallTargets(lst)], True)
    ///     return [op0, op1] + op2
    /// ```
    ///
    /// Rust adaptation: `op.args[0]` upstream is a `Ptr(FuncType)`; we
    /// synthesize that value via `OpKind::FuncptrFromVtable` on the
    /// `&dyn Trait` receiver (args[0]) so `GuardValue {kind='i'}` sees
    /// an integer function-address just like upstream (PRE-EXISTING-
    /// ADAPTATION cited at `jtransform.py:546`).
    fn handle_regular_indirect_call(
        &mut self,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        let (trait_root, method_name) = match target {
            CallTarget::Indirect {
                trait_root,
                method_name,
            } => (trait_root.clone(), method_name.clone()),
            // Called only from `rewrite_op_indirect_call`, which is
            // gated on `CallTarget::Indirect` above.
            _ => unreachable!("handle_regular_indirect_call expects CallTarget::Indirect"),
        };
        // Collect candidate jitcode indices via a mutable borrow
        // (`get_jitcode` may allocate a new index — RPython call.py:157-165).
        // jtransform.py:541-544: `for targetgraph in callcontrol.graphs_from(op)`.
        let cc_mut = self
            .callcontrol
            .as_deref_mut()
            .expect("handle_regular_indirect_call requires CallControl");
        let candidate_paths = cc_mut
            .graphs_from_indirect(&trait_root, &method_name)
            .unwrap_or_default();
        let all_paths = cc_mut.all_impls_for_indirect(&trait_root, &method_name);
        // call.py:259-280 family-wide validation.
        if let Err(err) = cc_mut.check_indirect_call_family(&all_paths) {
            panic!("rewrite_op_indirect_call ({trait_root}::{method_name}): {err}");
        }
        let jitcode_indices: Vec<usize> = candidate_paths
            .iter()
            .map(|p| cc_mut.get_jitcode(p))
            .collect();
        // Compute calldescr while we still hold `cc_mut`.
        let non_void_args = resolve_non_void_arg_types(args, self.type_state);
        let result_ir_type = value_type_to_ir_type(result_ty);
        let descriptor = cc_mut.getcalldescr(
            target,
            non_void_args,
            result_ir_type,
            OopSpecIndex::None,
            None,
            &mut self.analysis_cache,
        );
        // Pick the receiver variable (RPython `op.args[0]`).  In our
        // lowering the receiver is always args[0].
        let receiver = *args
            .first()
            .expect("dyn-Trait method call must have a receiver arg");
        // jtransform.py:545-552 emit sequence.
        let mut ops = Vec::<SpaceOperation>::new();
        // op0: SpaceOperation('-live-', ...)
        ops.push(SpaceOperation {
            result: None,
            kind: OpKind::Live,
        });
        // Rust adaptation: materialize the vtable entry as an int-typed
        // value, then apply `int_guard_value` to it.
        let funcptr_vid = self.allocate_synthetic_value();
        ops.push(SpaceOperation {
            result: Some(funcptr_vid),
            kind: OpKind::FuncptrFromVtable {
                receiver,
                trait_root: trait_root.clone(),
                method_name: method_name.clone(),
            },
        });
        // op1: SpaceOperation('int_guard_value', [op.args[0]], None)
        ops.push(SpaceOperation {
            result: None,
            kind: OpKind::GuardValue {
                value: funcptr_vid,
                kind_char: 'i',
            },
        });
        // op2: self.handle_residual_call(op, [IndirectCallTargets(lst)], True)
        let indirect_targets = Some(crate::model::IndirectCallTargets {
            candidates: jitcode_indices,
        });
        let residual = self.handle_residual_call_with_targets(
            op,
            target,
            descriptor,
            args,
            result_ty,
            graph_name,
            indirect_targets,
        );
        match residual {
            RewriteResult::Replace(tail) => {
                ops.extend(tail);
                RewriteResult::Replace(ops)
            }
            // `handle_residual_call_with_targets` always Replaces.
            other => other,
        }
    }

    /// RPython: elidable call — pure function, result depends only on args.
    /// RPython jtransform.py:546-562.
    ///
    /// `target` is the funcptr identity per jtransform.py:457.
    #[allow(dead_code)]
    fn handle_elidable_call(
        &mut self,
        op: &SpaceOperation,
        target: &CallTarget,
        descriptor: CallDescriptor,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("call {target} → elidable"),
        });
        self.calls_classified += 1;
        let (args_i, args_r, args_f) = self.make_three_lists(args);
        let result_kind = value_type_to_kind(result_ty);
        RewriteResult::Replace(vec![SpaceOperation {
            result: op.result,
            kind: OpKind::CallElidable {
                funcptr: target.clone(),
                descriptor,
                args_i,
                args_r,
                args_f,
                result_kind,
            },
        }])
    }

    /// RPython: may-force call — can trigger GC or force virtualizables.
    /// RPython jtransform.py:609-625.
    ///
    /// `target` is the funcptr identity per jtransform.py:457.
    #[allow(dead_code)]
    fn handle_may_force_call(
        &mut self,
        op: &SpaceOperation,
        target: &CallTarget,
        descriptor: CallDescriptor,
        args: &[ValueId],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("call {target} → may_force"),
        });
        self.calls_classified += 1;
        let (args_i, args_r, args_f) = self.make_three_lists(args);
        let result_kind = value_type_to_kind(result_ty);
        // RPython: call_may_force always followed by -live-
        RewriteResult::Replace(vec![
            SpaceOperation {
                result: op.result,
                kind: OpKind::CallMayForce {
                    funcptr: target.clone(),
                    descriptor,
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
}

/// RPython: `getkind(concretetype)[0]` → 'i', 'r', 'f', or 'v'.
///
/// RPython's rtyper resolves all types before jtransform runs, so
/// getkind() never sees an unknown type. In our pipeline, Unknown
/// means the annotate/rtype pass couldn't resolve the type. We map
/// Unknown to 'r' (ref) since most Python-level values are GC refs.
/// RPython: `NON_VOID_ARGS = [x.concretetype for x in op.args[1:]
///                             if x.concretetype is not Void]`
/// (call.py:220-221).
///
/// Resolve the IR types of call arguments, skipping Void.
fn resolve_non_void_arg_types(
    args: &[ValueId],
    type_state: Option<&crate::translate_legacy::rtyper::rtyper::TypeResolutionState>,
) -> Vec<majit_ir::value::Type> {
    args.iter()
        .filter_map(|&v| {
            let kind = if let Some(ts) = type_state {
                if let Some(ct) = ts.concrete_types.get(&v) {
                    match ct {
                        crate::translate_legacy::rtyper::rtyper::ConcreteType::Signed => 'i',
                        crate::translate_legacy::rtyper::rtyper::ConcreteType::GcRef => 'r',
                        crate::translate_legacy::rtyper::rtyper::ConcreteType::Float => 'f',
                        crate::translate_legacy::rtyper::rtyper::ConcreteType::Void => 'v',
                        crate::translate_legacy::rtyper::rtyper::ConcreteType::Unknown => 'r',
                    }
                } else {
                    'r'
                }
            } else {
                'r'
            };
            match kind {
                'v' => None, // RPython: skip Void args
                'i' => Some(majit_ir::value::Type::Int),
                'r' => Some(majit_ir::value::Type::Ref),
                'f' => Some(majit_ir::value::Type::Float),
                _ => Some(majit_ir::value::Type::Ref),
            }
        })
        .collect()
}

fn value_type_to_kind(ty: &ValueType) -> char {
    match ty {
        ValueType::Int | ValueType::State => 'i',
        ValueType::Ref | ValueType::Unknown => 'r',
        ValueType::Float => 'f',
        ValueType::Void => 'v',
    }
}

/// Convert codewriter ValueType to IR Type.
///
/// RPython: `x.concretetype` → lltype mapping.
/// Used by getcalldescr to build NON_VOID_ARGS and RESULT types.
fn value_type_to_ir_type(ty: &ValueType) -> majit_ir::value::Type {
    match ty {
        ValueType::Int | ValueType::State => majit_ir::value::Type::Int,
        ValueType::Ref | ValueType::Unknown => majit_ir::value::Type::Ref,
        ValueType::Float => majit_ir::value::Type::Float,
        ValueType::Void => majit_ir::value::Type::Void,
    }
}

/// Convert a CallTarget to a CallPath for jitcode lookup.
fn target_to_call_path(target: &CallTarget) -> crate::parse::CallPath {
    match target {
        CallTarget::FunctionPath { segments } => {
            crate::parse::CallPath::from_segments(segments.iter().map(String::as_str))
        }
        CallTarget::Method { name, .. } => crate::parse::CallPath::from_segments([name.as_str()]),
        // RPython: an indirect_call has no single jitcode-lookup path —
        // the family is handled via `graphs_from_indirect` + IndirectCallTargets.
        // This fallback returns a stub path only reached by callers that don't
        // distinguish; the real consumer (`handle_regular_indirect_call`) uses
        // the family path directly.
        CallTarget::Indirect {
            trait_root,
            method_name,
        } => crate::parse::CallPath::from_segments([trait_root.as_str(), method_name.as_str()]),
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

fn remap_list(
    values: &[ValueId],
    aliases: &std::collections::HashMap<ValueId, ValueId>,
) -> Vec<ValueId> {
    values
        .iter()
        .copied()
        .map(|v| remap_value(v, aliases))
        .collect()
}

fn remap_op(
    op: &SpaceOperation,
    aliases: &std::collections::HashMap<ValueId, ValueId>,
) -> SpaceOperation {
    let kind = match &op.kind {
        OpKind::Input { .. }
        | OpKind::ConstInt(_)
        | OpKind::VableForce
        | OpKind::CurrentTraceLength
        | OpKind::Live
        | OpKind::GuardValue { .. }
        | OpKind::FuncptrFromVtable { .. }
        | OpKind::Unknown { .. } => op.kind.clone(),
        OpKind::RecordQuasiImmutField {
            base,
            field,
            mutate_field,
        } => OpKind::RecordQuasiImmutField {
            base: remap_value(*base, aliases),
            field: field.clone(),
            mutate_field: mutate_field.clone(),
        },
        OpKind::FieldRead {
            base,
            field,
            ty,
            pure,
        } => OpKind::FieldRead {
            base: remap_value(*base, aliases),
            field: field.clone(),
            ty: ty.clone(),
            pure: *pure,
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
            array_type_id,
        } => OpKind::ArrayRead {
            base: remap_value(*base, aliases),
            index: remap_value(*index, aliases),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
        },
        OpKind::ArrayWrite {
            base,
            index,
            value,
            item_ty,
            array_type_id,
        } => OpKind::ArrayWrite {
            base: remap_value(*base, aliases),
            index: remap_value(*index, aliases),
            value: remap_value(*value, aliases),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
        },
        OpKind::InteriorFieldRead {
            base,
            index,
            field,
            item_ty,
            array_type_id,
        } => OpKind::InteriorFieldRead {
            base: remap_value(*base, aliases),
            index: remap_value(*index, aliases),
            field: field.clone(),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
        },
        OpKind::InteriorFieldWrite {
            base,
            index,
            field,
            value,
            item_ty,
            array_type_id,
        } => OpKind::InteriorFieldWrite {
            base: remap_value(*base, aliases),
            index: remap_value(*index, aliases),
            field: field.clone(),
            value: remap_value(*value, aliases),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
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
            array_itemsize,
            array_is_signed,
        } => OpKind::VableArrayRead {
            array_index: *array_index,
            elem_index: remap_value(*elem_index, aliases),
            item_ty: item_ty.clone(),
            array_itemsize: *array_itemsize,
            array_is_signed: *array_is_signed,
        },
        OpKind::VableArrayWrite {
            array_index,
            elem_index,
            value,
            item_ty,
            array_itemsize,
            array_is_signed,
        } => OpKind::VableArrayWrite {
            array_index: *array_index,
            elem_index: remap_value(*elem_index, aliases),
            value: remap_value(*value, aliases),
            item_ty: item_ty.clone(),
            array_itemsize: *array_itemsize,
            array_is_signed: *array_is_signed,
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
        OpKind::JitDebug { args } => OpKind::JitDebug {
            args: remap_list(args, aliases),
        },
        OpKind::RecordKnownResult {
            result_value,
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => OpKind::RecordKnownResult {
            result_value: remap_value(*result_value, aliases),
            funcptr: funcptr.clone(),
            descriptor: descriptor.clone(),
            args_i: remap_list(args_i, aliases),
            args_r: remap_list(args_r, aliases),
            args_f: remap_list(args_f, aliases),
            result_kind: *result_kind,
        },
        OpKind::AssertGreen { value, kind_char } => OpKind::AssertGreen {
            value: remap_value(*value, aliases),
            kind_char: *kind_char,
        },
        OpKind::IsConstant { value, kind_char } => OpKind::IsConstant {
            value: remap_value(*value, aliases),
            kind_char: *kind_char,
        },
        OpKind::IsVirtual { value, kind_char } => OpKind::IsVirtual {
            value: remap_value(*value, aliases),
            kind_char: *kind_char,
        },
        OpKind::CallElidable {
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => OpKind::CallElidable {
            funcptr: funcptr.clone(),
            descriptor: descriptor.clone(),
            args_i: remap_list(args_i, aliases),
            args_r: remap_list(args_r, aliases),
            args_f: remap_list(args_f, aliases),
            result_kind: *result_kind,
        },
        OpKind::CallResidual {
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
            indirect_targets,
        } => OpKind::CallResidual {
            funcptr: funcptr.clone(),
            descriptor: descriptor.clone(),
            args_i: remap_list(args_i, aliases),
            args_r: remap_list(args_r, aliases),
            args_f: remap_list(args_f, aliases),
            indirect_targets: indirect_targets.clone(),
            result_kind: *result_kind,
        },
        OpKind::CallMayForce {
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => OpKind::CallMayForce {
            funcptr: funcptr.clone(),
            descriptor: descriptor.clone(),
            args_i: remap_list(args_i, aliases),
            args_r: remap_list(args_r, aliases),
            args_f: remap_list(args_f, aliases),
            result_kind: *result_kind,
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
        OpKind::ConditionalCall {
            condition,
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
        } => OpKind::ConditionalCall {
            condition: remap_value(*condition, aliases),
            funcptr: funcptr.clone(),
            descriptor: descriptor.clone(),
            args_i: remap_list(args_i, aliases),
            args_r: remap_list(args_r, aliases),
            args_f: remap_list(args_f, aliases),
        },
        OpKind::ConditionalCallValue {
            value,
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => OpKind::ConditionalCallValue {
            value: remap_value(*value, aliases),
            funcptr: funcptr.clone(),
            descriptor: descriptor.clone(),
            args_i: remap_list(args_i, aliases),
            args_r: remap_list(args_r, aliases),
            args_f: remap_list(args_f, aliases),
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

/// Map a user-level oopspec string (from `@oopspec(...)`) to an `OopSpecIndex`.
///
/// rlib/jit.py:250 — `@oopspec(spec)` stores a spec string on the function.
/// jtransform.py:1731-1755 `__handle_jit_call` patterns the spec name.
///
/// For the JIT-specific `jit.*` specs, RPython emits SpaceOperations with
/// distinct names (e.g. `jit_debug`, `int_isconstant`); for list/dict/str
/// specs RPython uses dedicated OS_* indices. This helper currently maps
/// the cases that have a direct OopSpecIndex equivalent.
fn map_user_oopspec_to_index(spec: &str) -> majit_ir::descr::OopSpecIndex {
    use majit_ir::descr::OopSpecIndex;
    // Normalize: `jit.isconstant(value)` → `jit.isconstant`
    let base = spec.split('(').next().unwrap_or(spec).trim();
    match base {
        // All jit.* oopspecs are intercepted by _handle_jit_call() before
        // reaching this function. Remaining oopspecs map to OS_* indices.
        "virtual_ref" | "virtual_ref_finish" => OopSpecIndex::JitForceVirtualizable,
        _ => OopSpecIndex::None,
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
        if info.check_forces_virtual_or_virtualizable() {
            CallEffectKind::MayForce
        } else if info.check_is_elidable() {
            CallEffectKind::Elidable
        } else {
            CallEffectKind::Residual
        }
    }

    if let Some(descriptor) = overrides
        .iter()
        .find(|override_| call_target_matches_loose(&override_.target, target))
        .map(|override_| override_.descriptor.clone())
    {
        let effect = classify_effect_info(&descriptor.get_extra_info());
        return Some((descriptor, effect));
    }
    let descriptor = crate::call::describe_call(target)?;
    let effect = classify_effect_info(&descriptor.get_extra_info());
    Some((descriptor, effect))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{CallTarget, FunctionGraph, OpKind, ValueType};

    #[test]
    fn rewrite_graph_tags_vable_fields() {
        let mut graph = FunctionGraph::new("test");
        let base = graph.alloc_value();
        graph.push_op(
            graph.startblock,
            OpKind::FieldRead {
                base,
                field: crate::model::FieldDescriptor::new("next_instr", Some("Frame".into())),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        graph.set_terminator(graph.startblock, Terminator::Return(None));

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
        let rewritten_op = &result.graph.block(graph.startblock).operations[0];
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
            graph.startblock,
            OpKind::FieldRead {
                base,
                field: crate::model::FieldDescriptor::new("next_instr", Some("OtherFrame".into())),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        graph.set_terminator(graph.startblock, Terminator::Return(None));

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
            result.graph.block(graph.startblock).operations[0].kind,
            OpKind::FieldRead { .. }
        ));
    }

    #[test]
    fn rewrite_graph_classifies_calls() {
        let mut graph = FunctionGraph::new("test");
        graph.push_op(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::method("call_callable", Some("PyFrame".into())),
                args: vec![],
                result_ty: ValueType::Ref,
            },
            false,
        );
        graph.set_terminator(graph.startblock, Terminator::Return(None));

        let result = rewrite_graph(
            &graph,
            &crate::test_support::pyre_pipeline_config().transform,
        );
        assert_eq!(result.calls_classified, 1);
        assert!(matches!(
            result.graph.block(graph.startblock).operations[0].kind,
            OpKind::CallResidual { .. }
        ));
    }

    #[test]
    fn rewrite_graph_uses_explicit_call_effect_overrides() {
        // RPython: residual calls always produce residual_call_*, regardless
        // of effect. The effect is only in the calldescr (descriptor).
        let mut graph = FunctionGraph::new("test");
        graph.push_op(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["custom_reader"]),
                args: vec![],
                result_ty: ValueType::Ref,
            },
            true,
        );
        graph.set_terminator(graph.startblock, Terminator::Return(None));

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
        // RPython: always residual_call_*, effect in descriptor only.
        assert!(matches!(
            result.graph.block(graph.startblock).operations[0].kind,
            OpKind::CallResidual { .. }
        ));
        // Verify the effect is correctly carried in the descriptor.
        if let OpKind::CallResidual { descriptor, .. } =
            &result.graph.block(graph.startblock).operations[0].kind
        {
            assert!(
                descriptor
                    .extra_info
                    .check_forces_virtual_or_virtualizable()
            );
        } else {
            panic!("expected CallResidual");
        }
    }

    #[test]
    fn rewrite_graph_reports_unknowns() {
        let mut graph = FunctionGraph::new("demo");
        graph.push_op(
            graph.startblock,
            OpKind::Unknown {
                kind: crate::model::UnknownKind::UnsupportedExpr,
            },
            false,
        );
        graph.set_terminator(
            graph.startblock,
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
        graph.block_mut(graph.startblock).inputargs.push(frame);
        graph.push_op(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["hint_access_directly"]),
                args: vec![frame],
                result_ty: ValueType::Ref,
            },
            false,
        );
        graph
            .block_mut(graph.startblock)
            .operations
            .last_mut()
            .unwrap()
            .result = Some(hinted);
        graph.push_op(
            graph.startblock,
            OpKind::FieldRead {
                base: hinted,
                field: crate::model::FieldDescriptor::new("next_instr", Some("Frame".into())),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        graph.set_terminator(graph.startblock, Terminator::Return(None));

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

        assert_eq!(result.graph.block(graph.startblock).operations.len(), 1);
        match &result.graph.block(graph.startblock).operations[0].kind {
            OpKind::VableFieldRead { field_index, .. } => assert_eq!(*field_index, 0),
            other => panic!("expected VableFieldRead after hint suppression, got {other:?}"),
        }
    }

    #[test]
    fn rewrite_graph_rewrites_hint_force_virtualizable() {
        let mut graph = FunctionGraph::new("demo");
        let frame = graph.alloc_value();
        let forced = graph.alloc_value();
        graph.block_mut(graph.startblock).inputargs.push(frame);
        graph.push_op(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["hint_force_virtualizable"]),
                args: vec![frame],
                result_ty: ValueType::Ref,
            },
            false,
        );
        graph
            .block_mut(graph.startblock)
            .operations
            .last_mut()
            .unwrap()
            .result = Some(forced);
        graph.push_op(
            graph.startblock,
            OpKind::FieldRead {
                base: forced,
                field: crate::model::FieldDescriptor::new("next_instr", Some("Frame".into())),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        graph.set_terminator(graph.startblock, Terminator::Return(None));

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

        let ops = &result.graph.block(graph.startblock).operations;
        assert!(matches!(ops[0].kind, OpKind::VableForce));
        assert!(matches!(
            ops[1].kind,
            OpKind::VableFieldRead { field_index: 0, .. }
        ));
    }

    // ── RPython indirect_call plumbing tests — plan §Tests ──────────

    /// Full pipeline: build a graph that calls `receiver.run()` on a
    /// `dyn Handler` receiver with two registered impls, run the
    /// transformer, and assert the emitted sequence is
    /// `[Live, FuncptrFromVtable, GuardValue{kind='i'}, CallResidual{indirect_targets=Some}, Live]`.
    /// RPython `jtransform.py:538-553` port parity.
    #[test]
    fn handle_regular_indirect_call_emit_order() {
        use crate::call::CallControl;
        use crate::model::FunctionGraph as CrateFG;

        let mut cc = CallControl::new();
        cc.register_trait_method("run", Some("Handler"), "A", CrateFG::new("A::run"));
        cc.register_trait_method("run", Some("Handler"), "B", CrateFG::new("B::run"));
        cc.find_all_graphs_for_tests();

        // Build a graph: `receiver = input(); receiver.run()`.
        let mut graph = FunctionGraph::new("outer");
        let receiver = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "handler".to_string(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.push_op(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::indirect("Handler", "run"),
                args: vec![receiver],
                result_ty: ValueType::Void,
            },
            true,
        );
        graph.set_terminator(graph.startblock, Terminator::Return(None));

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
        let result = transformer.transform(&graph);

        let ops = &result.graph.block(graph.startblock).operations;
        // Strip the initial Input op.
        let post_input = &ops[1..];
        assert!(
            matches!(post_input[0].kind, OpKind::Live),
            "expected Live first, got {:?}",
            post_input[0].kind
        );
        assert!(
            matches!(post_input[1].kind, OpKind::FuncptrFromVtable { .. }),
            "expected FuncptrFromVtable second, got {:?}",
            post_input[1].kind
        );
        match &post_input[2].kind {
            OpKind::GuardValue { kind_char: 'i', .. } => {}
            other => panic!("expected GuardValue kind='i', got {other:?}"),
        }
        match &post_input[3].kind {
            OpKind::CallResidual {
                indirect_targets: Some(t),
                ..
            } => assert_eq!(t.candidates.len(), 2, "both impls should be candidates"),
            other => panic!("expected CallResidual with indirect_targets, got {other:?}"),
        }
        // jtransform.py:547 handle_residual_call(..., may_call_jitcodes=True)
        // forces a trailing `-live-`.
        assert!(
            matches!(post_input[4].kind, OpKind::Live),
            "expected trailing Live, got {:?}",
            post_input[4].kind
        );
    }

    /// End-to-end smoke: after `rewrite_op_indirect_call`, the
    /// `CallResidual.indirect_targets` payload carries exactly one
    /// jitcode index per candidate impl (assigned by
    /// `CallControl::get_jitcode`).  This is what the assembler later
    /// merges into `Assembler.indirectcalltargets` (`assembler.py:208-209`).
    #[test]
    fn indirectcalltargets_reach_call_residual_payload() {
        use crate::call::CallControl;
        use crate::model::FunctionGraph as CrateFG;

        let mut cc = CallControl::new();
        cc.register_trait_method("run", Some("Handler"), "A", CrateFG::new("A::run"));
        cc.register_trait_method("run", Some("Handler"), "B", CrateFG::new("B::run"));
        cc.find_all_graphs_for_tests();

        let mut graph = FunctionGraph::new("outer");
        let receiver = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "handler".to_string(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.push_op(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::indirect("Handler", "run"),
                args: vec![receiver],
                result_ty: ValueType::Void,
            },
            true,
        );
        graph.set_terminator(graph.startblock, Terminator::Return(None));

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
        let result = transformer.transform(&graph);

        let residual = result
            .graph
            .block(graph.startblock)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::CallResidual {
                    indirect_targets, ..
                } => indirect_targets.clone(),
                _ => None,
            })
            .expect("residual call with indirect_targets");

        assert_eq!(residual.candidates.len(), 2);
        // Jitcode indices are sequentially assigned from 0.
        let mut sorted = residual.candidates.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1]);
    }

    /// RPython `rpython/jit/codewriter/jtransform.py:895-903` — a
    /// quasi-immutable field read lowers to
    /// `[-live-, record_quasiimmut_field(v, descr, descr1), getfield_*_pure]`.
    /// Covers Issue 5.
    #[test]
    fn getfield_rewrite_emits_record_quasiimmut_for_quasi_immut() {
        use crate::call::CallControl;
        use crate::model::{FieldDescriptor, ImmutableRank};

        let mut cc = CallControl::new();
        cc.immutable_fields_by_struct.insert(
            "Cell".to_string(),
            vec![("value".to_string(), ImmutableRank::QuasiImmutable)],
        );

        let mut graph = FunctionGraph::new("read_cell");
        let base = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "cell".to_string(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        graph.push_op(
            graph.startblock,
            OpKind::FieldRead {
                base,
                field: FieldDescriptor::new("value", Some("Cell".to_string())),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        graph.set_terminator(graph.startblock, Terminator::Return(None));

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
        let result = transformer.transform(&graph);
        let ops: Vec<&OpKind> = result
            .graph
            .block(graph.startblock)
            .operations
            .iter()
            .map(|o| &o.kind)
            .collect();

        // Expect the triple [Live, RecordQuasiImmutField, FieldRead] in
        // order, preceded by the Input op.
        let live_idx = ops
            .iter()
            .position(|k| matches!(k, OpKind::Live))
            .expect("Live marker present");
        assert!(matches!(
            ops[live_idx + 1],
            OpKind::RecordQuasiImmutField {
                field, mutate_field, ..
            } if field.name == "value"
                && mutate_field.name == "mutate_value"
                && mutate_field.owner_root.as_deref() == Some("Cell")
        ));
        assert!(matches!(
            ops[live_idx + 2],
            OpKind::FieldRead {
                field, pure: true, ..
            } if field.name == "value"
        ));
    }

    /// A plain-immutable field read lowers directly to a pure read, without
    /// the quasi-immutable bookkeeping pair.  Mirrors the `pure` /
    /// non-`pure` fork at `jtransform.py:867-878`.
    #[test]
    fn getfield_rewrite_preserves_plain_immutable_read() {
        use crate::call::CallControl;
        use crate::model::{FieldDescriptor, ImmutableRank};

        let mut cc = CallControl::new();
        cc.immutable_fields_by_struct.insert(
            "Point".to_string(),
            vec![("x".to_string(), ImmutableRank::Immutable)],
        );

        let mut graph = FunctionGraph::new("read_x");
        let base = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "p".to_string(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        graph.push_op(
            graph.startblock,
            OpKind::FieldRead {
                base,
                field: FieldDescriptor::new("x", Some("Point".to_string())),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        graph.set_terminator(graph.startblock, Terminator::Return(None));

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
        let result = transformer.transform(&graph);
        let ops = &result.graph.block(graph.startblock).operations;

        assert!(
            !ops.iter()
                .any(|o| matches!(o.kind, OpKind::RecordQuasiImmutField { .. })),
            "plain immutable must not emit record_quasiimmut_field"
        );
        assert!(
            ops.iter().any(|o| matches!(
                &o.kind,
                OpKind::FieldRead {
                    field, pure: true, ..
                } if field.name == "x"
            )),
            "FieldRead for x should become a pure read"
        );
    }
}
