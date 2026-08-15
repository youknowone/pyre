//! Graph-based jtransform: semantic rewrite pass.
#![allow(non_snake_case)]
//!
//! RPython equivalent: jtransform.py Transformer.optimize_block()
//!
//! Transforms a FunctionGraph by rewriting operations:
//! - FieldRead on virtualizable fields → VableFieldRead marker
//! - FieldWrite on virtualizable fields → VableFieldWrite marker
//! - ArrayRead on virtualizable arrays → VableArrayRead marker
//! - Call classification → elidable/residual/may_force tagging

use std::collections::BTreeMap;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::call::CallDescriptor;
use crate::codewriter::support::{NormalizedArg, decode_builtin_call};
use crate::model::{
    CallFuncPtr, CallTarget, FieldDescriptor, FunctionGraph, LinkArg, OpKind, SpaceOperation,
    ValueType, remap_control_flow_metadata_var,
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
        if self.name != field.name {
            return false;
        }
        // An owner-less config entry must match nothing rather than every
        // struct's field of that name.  A leaf name does not identify a
        // type here — the corpus carries `FrameBlock.valuestackdepth`
        // alongside the `PyFrame.valuestackdepth` this config declares
        // virtualizable — so a name-only match would lower an unrelated
        // struct's field through the virtualizable protocol.  Failing
        // closed costs at worst a missed lowering (a plain `getfield`,
        // which is correct if slower); failing open is wrong code.
        // No producer builds one: every `VirtualizableFieldDescriptor`
        // constructor call in the tree passes an owner.
        let Some(cfg_owner) = self.owner_root.as_ref() else {
            return false;
        };
        let Some(field_owner) = field.owner_root.as_ref() else {
            return false;
        };
        // RPython `jtransform.py:982 self.callcontrol.get_vinfo(
        // v_virtualizable.concretetype)` compares lltype object IDENTITY
        // — a single `Ptr(GcStruct(...))` instance per type, looked up
        // by the call control's `vinfos` dict.  Pyre carries the same
        // identity as a string `owner_root`; the canonical form is
        // `"<module_path>::<bare_name>"` (see
        // `descr.rs canonical_struct_name`).  Helpers built from
        // `impl OpcodeStepExecutor for PyFrame` in `pyre-interpreter`
        // resolve `self` to `"pyframe::PyFrame"`; the vable config
        // (`virtualizable_spec::PYFRAME_VABLE_OWNER_ROOT`) ships the
        // bare `"PyFrame"` for symmetry with `lib.rs`'s fixture-side
        // vable_fields registrations.  Compare against the canonical
        // form so both shapes match without mutating either source of
        // truth.
        if cfg_owner == field_owner {
            return true;
        }
        let cfg_canonical = majit_ir::descr::canonical_struct_name(cfg_owner);
        let field_canonical = majit_ir::descr::canonical_struct_name(field_owner);
        cfg_canonical == field_canonical
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
    /// A callee whose effects row is stated by the override table rather than
    /// derived from a graph (callee census) — upstream's `analyze_external_call` answer
    /// (`graphanalyze.py:104-108`).
    ///
    /// The three variants above are shorthands for three particular rows;
    /// this one carries the row.
    Declared(DeclaredCallEffects),
}

/// `effectinfo.py:17-24` `EF_*`, minus `EF_RANDOM_EFFECTS`.
///
/// The omission is the enforcement. `EF_RANDOM_EFFECTS` is top — the answer
/// for a callee nobody can describe — so it is not something a declaration
/// can say, and leaving it out makes upstream's
/// `assert not (elidable_function and random_effects_on_gcobjs)`
/// (`rpython/rtyper/lltypesystem/rffi.py:160`) hold by construction. The
/// pairing that assert forbids cannot be spelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeclaredExtraEffect {
    ElidableCannotRaise,
    LoopInvariant,
    CannotRaise,
    ElidableOrMemoryError,
    ElidableCanRaise,
    CanRaise,
    ForcesVirtualOrVirtualizable,
}

impl From<DeclaredExtraEffect> for ExtraEffect {
    fn from(declared: DeclaredExtraEffect) -> Self {
        match declared {
            DeclaredExtraEffect::ElidableCannotRaise => ExtraEffect::ElidableCannotRaise,
            DeclaredExtraEffect::LoopInvariant => ExtraEffect::LoopInvariant,
            DeclaredExtraEffect::CannotRaise => ExtraEffect::CannotRaise,
            DeclaredExtraEffect::ElidableOrMemoryError => ExtraEffect::ElidableOrMemoryError,
            DeclaredExtraEffect::ElidableCanRaise => ExtraEffect::ElidableCanRaise,
            DeclaredExtraEffect::CanRaise => ExtraEffect::CanRaise,
            DeclaredExtraEffect::ForcesVirtualOrVirtualizable => {
                ExtraEffect::ForcesVirtualOrVirtualizable
            }
        }
    }
}

/// One declared effects row.
///
/// The six read/write descr sets stay at `EffectInfo`'s default (empty).
/// The override table is built before the descr universe exists, so it has no
/// way to name a descr. Empty is the right row for a callee that touches no
/// field or array and the wrong row for anything else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeclaredCallEffects {
    pub extra: DeclaredExtraEffect,
    /// `effectinfo.py:125 can_collect`.
    pub can_collect: bool,
    pub can_invalidate: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallEffectOverride {
    /// `op.args[0]`-equivalent funcptr identity used to match the
    /// override against a call site.
    ///
    /// Spell this the way the **resolver** spells it at the call site, which
    /// is not necessarily a name the source artifact declares. Reading the
    /// `.ullbc` cannot tell you: it registers `opcode_binary_op` under both a
    /// bare and a module-qualified name, and every one of the 38 opcode call
    /// sites resolves to the qualified `["pyre_interpreter", "pyopcode",
    /// "opcode_binary_op"]`. A row written with the bare name is legitimate
    /// against the artifact and still matches nothing, because non-`Method`
    /// patterns are compared by full structural equality.
    ///
    /// The spelling a call site actually carries is what `PYRE_CALLEE_CENSUS=1`
    /// prints in its `with_graph` table (set `PYRE_CALLEE_CENSUS_ROWS=all`;
    /// the default cap is 25 rows). Confirm a new row against the per-entry
    /// match counter in the same output — a row that reads `0x INERT` is not
    /// installed, whatever it looks like here.
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
        CallEffectKind::Declared(declared) => EffectInfo {
            can_collect: declared.can_collect,
            can_invalidate: declared.can_invalidate,
            ..EffectInfo::new(declared.extra.into(), OopSpecIndex::None)
        },
    }
}

/// A note about a transformation applied to the graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphTransformNote {
    pub function: String,
    pub detail: String,
}

/// Result of a graph transformation pass.
#[derive(Debug, Clone)]
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
///
/// This wrapper does NOT run `lower_indirect_calls` (the
/// rtyper-equivalent pass lives in
/// `translator/rtyper/rpbc.rs`).  Callers that can produce
/// `CallTarget::Indirect` must go through
/// `codewriter::transform_graph_to_jitcode` instead, which threads
/// `&CallControl` and runs the lowering pass before jtransform.
/// This debug-assertion catches missed lowering sites at the
/// remaining entries (test fixtures).
pub fn transform_graph(
    graph: &FunctionGraph,
    config: &GraphTransformConfig,
) -> GraphTransformResult {
    #[cfg(debug_assertions)]
    crate::translator::rtyper::rpbc::assert_no_indirect_call_targets(graph);
    let mut transformer = Transformer::new(config);
    transformer.transform(graph)
}

/// `jtransform.py:53-57 integer_bounds(size, unsigned)`.
pub fn integer_bounds(size: usize, unsigned: bool) -> (i128, i128) {
    if unsigned {
        (0, 1_i128 << (8 * size))
    } else {
        (-(1_i128 << (8 * size - 1)), 1_i128 << (8 * size - 1))
    }
}

/// `jtransform.py:2276-2277 keep_operation_unchanged(jtransform, op)`.
pub fn keep_operation_unchanged(
    _jtransform: &Transformer<'_>,
    op: &SpaceOperation,
) -> SpaceOperation {
    op.clone()
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
    /// RPython: `Transformer.portal_jd` (`jtransform.py:65`) —
    /// "non-None only for the portal graph(s)". Consulted by
    /// `handle_jit_marker__jit_merge_point` (`jtransform.py:1690-1712`)
    /// to stamp `portal_jd.index` onto the rewritten op and to assert
    /// the marker's jitdriver matches this portal's. Pyre stores the
    /// index alone; the full `JitDriverStaticData` is owned by
    /// `CallControl::jitdrivers_sd` and can be looked up by index there.
    portal_jd_index: Option<usize>,
    /// RPython: `Transformer.__init__` config for virtualizable lowering.
    config: &'a GraphTransformConfig,
    /// Type resolution state from the rtype pass.
    /// Used by `make_three_lists()` to split args by kind.
    /// RPython: types are on `Variable.concretetype` — we pass them explicitly.
    /// RPython: `Transformer.vable_array_vars`.
    /// Stores (vable_base, array_index, itemsize, is_signed) per vable array variable.
    vable_array_vars: std::collections::HashMap<
        crate::flowspace::model::Variable,
        (crate::flowspace::model::Variable, usize, usize, bool),
    >,
    /// RPython: `Transformer.vable_flags`. Keyed by Variable identity
    /// matching upstream `self.vable_flags[op.args[0]] = ...`
    /// (`jtransform.py` populates with `Variable` objects).
    vable_flags: std::collections::HashMap<crate::flowspace::model::Variable, VableFlag>,
    /// Value aliases from identity rewrites (same_as / hint rewriting).
    aliases: std::collections::HashMap<
        crate::flowspace::model::Variable,
        crate::flowspace::model::Variable,
    >,
    notes: Vec<GraphTransformNote>,
    vable_rewrites: usize,
    calls_classified: usize,
    /// RPython: DependencyTracker — caches transitive analysis results.
    /// Shared across all getcalldescr() calls within this transform pass.
    analysis_cache: crate::call::AnalysisCache,
    /// Results consumed as explicit arguments by a direct or indirect call in
    /// the graph before call lowering rewrites those operations.
    call_argument_vars: Vec<crate::flowspace::model::Variable>,
}

/// RPython: jtransform.py vable_flags values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VableFlag {
    FreshVirtualizable,
}

/// `support.py:723 result.append(Constant(obj, lltype.typeOf(obj)))`
/// — prepend the constant-injection ops that `decode_builtin_call`'s
/// `NormalizedArg::ConstInt(v)` slots required, in front of the
/// downstream rewrite result.  No-op when no ConstInt slot was
/// materialised.  `Identity` / `Keep` are passed through unchanged
/// (downstream consumers of those variants do not observe the
/// materialised Variables — only `Replace` branches receive the
/// `effective_args` list).
fn prepend_const_prefix(prefix: &mut Vec<SpaceOperation>, result: RewriteResult) -> RewriteResult {
    if prefix.is_empty() {
        return result;
    }
    match result {
        RewriteResult::Replace(ops) => {
            let mut combined = std::mem::take(prefix);
            combined.extend(ops);
            RewriteResult::Replace(combined)
        }
        other => other,
    }
}

/// Result of rewriting a single operation.
///
/// RPython: `rewrite_operation()` returns SpaceOperation, list, None, or Constant.
enum RewriteResult {
    /// Replace with these ops
    Replace(Vec<SpaceOperation>),
    /// Remove the op (identity/alias: result remaps to the given Variable).
    /// RPython `jtransform.py:236 rewrite_op_same_as` returns `op.args[0]`
    /// directly; pyre wraps it in this enum so the caller can fold the
    /// alias into the rename map.
    Identity(crate::flowspace::model::Variable),
    /// Keep the op unchanged
    Keep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ResolvedCallResult {
    kind: char,
    ir_type: majit_ir::value::Type,
}

/// RPython: the `key` value stored as `op.args[0]` of a
/// `SpaceOperation('jit_marker', [key, jitdriver, *args])` operation
/// (`jtransform.py:1658-1663`). pyre's front-end does not carry the
/// `jit_marker` opname explicitly — the markers reach the codewriter as
/// `direct_call` s to `PyPyJitDriver::{jit_merge_point, can_enter_jit,
/// loop_header}`. This enum keeps the upstream key distinction inside
/// the dispatch hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JitMarkerKey {
    JitMergePoint,
    /// `can_enter_jit` aliases to `handle_jit_marker__loop_header`
    /// (jtransform.py:1723).
    CanEnterJit,
    LoopHeader,
}

/// JitDriver receiver types whose `jit_merge_point`/`can_enter_jit`/`loop_header` markers are recognized; each becomes its own portal via `portal_jd_index`.
const RECOGNIZED_JITDRIVER_RECEIVER_ROOTS: &[&str] = &["PyPyJitDriver", "UnpackIterableJitDriver"];

/// The null-pointer builtins `HostEnv::bootstrap` registers
/// (`flowspace/model.rs`), spelled as `HostObject::qualname`.
const NULL_PTR_BUILTIN_QUALNAMES: &[&str] = &[
    "core.ptr.null",
    "core.ptr.null_mut",
    "std.ptr.null",
    "std.ptr.null_mut",
];

/// Whether a 0-arg `FunctionPath` call names one of those builtins.
///
/// A guest const-global defined as `std::ptr::null_mut()` is bound to the SAME
/// `HostObject` as the spelled-out call (`flowspace/model.rs` binds the attr to
/// `core.ptr.null_mut` rather than minting a second callable), which is how
/// `flowspace_adapter`'s Branch 3b types the two identically. The surviving
/// model graph still carries whichever path the source spelled, so resolve it
/// through that same registry instead of listing the guest's spellings here —
/// the alias keeps one owner, and no guest name is repeated in the codewriter.
fn resolves_to_null_ptr_builtin(segments: &[String]) -> bool {
    let Some((leaf, module_path)) = segments.split_last() else {
        return false;
    };
    if module_path.is_empty() {
        return false;
    }
    crate::flowspace::model::HOST_ENV
        .import_module(&module_path.join("."))
        .and_then(|module| module.module_get(leaf))
        .is_some_and(|attr| NULL_PTR_BUILTIN_QUALNAMES.contains(&attr.qualname()))
}

fn jit_marker_key_from_target(target: &CallTarget) -> Option<JitMarkerKey> {
    let CallTarget::Method {
        name,
        receiver_root: Some(receiver_root),
        ..
    } = target
    else {
        return None;
    };
    if !RECOGNIZED_JITDRIVER_RECEIVER_ROOTS.contains(&receiver_root.as_str()) {
        return None;
    }
    match name.as_str() {
        "jit_merge_point" => Some(JitMarkerKey::JitMergePoint),
        "can_enter_jit" => Some(JitMarkerKey::CanEnterJit),
        "loop_header" => Some(JitMarkerKey::LoopHeader),
        _ => None,
    }
}

/// Split a run of [`Variable`]s into (ints, refs, floats) per upstream
/// `make_three_lists` (`jtransform.py:1616-1627`). Void values are
/// dropped, matching the upstream filter; Unknown defaults to `Ref`.
/// Reads kinds via `FunctionGraph::concretetype_of(var)` — the same
/// `getkind(v.concretetype)` source as RPython's `flatten.py:382
/// getcolor` and `rtyper.py:258 v.concretetype = ...`.
fn split_args_by_kind(
    args: &[crate::flowspace::model::Variable],
) -> (
    Vec<crate::flowspace::model::Variable>,
    Vec<crate::flowspace::model::Variable>,
    Vec<crate::flowspace::model::Variable>,
) {
    let mut ints = Vec::new();
    let mut refs = Vec::new();
    let mut floats = Vec::new();
    for v in args {
        let kind = match FunctionGraph::concretetype_of(v) {
            crate::codewriter::type_state::ConcreteType::Signed => 'i',
            crate::codewriter::type_state::ConcreteType::Float => 'f',
            crate::codewriter::type_state::ConcreteType::Void => 'v',
            // RPython: GcRef or Unknown → 'r'
            crate::codewriter::type_state::ConcreteType::GcRef
            | crate::codewriter::type_state::ConcreteType::Unknown => 'r',
        };
        match kind {
            'i' => ints.push(v.clone()),
            'f' => floats.push(v.clone()),
            'v' => {}
            _ => refs.push(v.clone()),
        }
    }
    (ints, refs, floats)
}

/// Pyre's frontend materialises source constants as SSA Variables. Recover
/// the upstream `Constant`/`Variable` distinction at the graph boundary.
fn is_source_constant_variable(
    graph: &FunctionGraph,
    variable: &crate::flowspace::model::Variable,
) -> bool {
    graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            if op.result.as_ref() != Some(variable) {
                return false;
            }
            match &op.kind {
                OpKind::ConstInt(_)
                | OpKind::ConstBool(_)
                | OpKind::ConstFloat(_)
                | OpKind::ConstStr(_)
                | OpKind::ConstRef(_)
                | OpKind::ConstRefNull
                | OpKind::ConstNone
                | OpKind::ConstRefAddr(_) => true,
                OpKind::Call { target, .. } => match target {
                    CallTarget::FunctionPath { segments } => {
                        segments.first().is_some_and(|s| s == "__str_const")
                            || crate::model::fn_const_segments(target).is_some()
                    }
                    _ => false,
                },
                _ => false,
            }
        })
    })
}

/// RPython: `support.autodetect_jit_markers_redvars` (support.py:64-102).
/// Compute the Variables live across the portal's `jit_merge_point`, remove
/// its greens, filter Constants/Void, and return INT/REF/FLOAT order.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn autodetect_jit_markers_redvars(
    graph: &FunctionGraph,
    greens: &[crate::flowspace::model::Variable],
) -> Vec<crate::flowspace::model::Variable> {
    use crate::codewriter::type_state::ConcreteType;
    use crate::flowspace::model::Variable;

    let (block, marker_index, marker_receiver, marker_greens) = graph
        .blocks
        .iter()
        .find_map(|block| {
            block.operations.iter().enumerate().find_map(|(index, op)| {
                let OpKind::Call { target, args, .. } = &op.kind else {
                    return None;
                };
                (jit_marker_key_from_target(target) == Some(JitMarkerKey::JitMergePoint)).then(
                    || {
                        (
                            block,
                            index,
                            args.first().cloned(),
                            args.iter()
                                .skip(1)
                                .take(greens.len())
                                .cloned()
                                .collect::<Vec<_>>(),
                        )
                    },
                )
            })
        })
        .expect("autoreds portal graph must contain a jit_merge_point");

    let mut alive_v: std::collections::HashSet<Variable> = std::collections::HashSet::new();
    for link in &block.exits {
        alive_v.extend(
            link.args
                .iter()
                .filter_map(|arg| arg.as_variable())
                .cloned(),
        );
        for extra in [&link.last_exception, &link.last_exc_value] {
            if let Some(variable) = extra.as_ref().and_then(|arg| arg.as_variable()) {
                alive_v.remove(variable);
            }
        }
    }
    for op in block.operations[marker_index + 1..].iter().rev() {
        if let Some(result) = &op.result {
            alive_v.remove(result);
        }
        alive_v.extend(crate::inline::op_variable_refs(&op.kind));
    }
    for green in greens {
        alive_v.remove(green);
    }
    // RPython subtracts `op.args[2:]` from the same graph operation it
    // scanned. During pyre's in-place block rewrite `greens` can already be
    // alias-remapped while the links still carry graph-resident identities,
    // so subtract that exact marker payload as the canonical green set.
    for green in &marker_greens {
        alive_v.remove(green);
    }
    // RPython's `op.args[1]` is the JitDriver Constant. Pyre's Rust frontend
    // materialises the singleton receiver as a Variable and can thread it
    // through loop links, so remove that exact marker receiver identity.
    if let Some(receiver) = &marker_receiver {
        alive_v.remove(receiver);
    }

    let mut reds_v: Vec<Variable> = alive_v
        .into_iter()
        .filter(|variable| {
            FunctionGraph::concretetype_of(variable) != ConcreteType::Void
                && !is_source_constant_variable(graph, variable)
        })
        .collect();
    // RPython's `sort_vars` is stable within each kind because its set order
    // is stable for a translation. Rust HashSet iteration is randomized, so
    // use Variable identity as a deterministic secondary key; otherwise the
    // emitted jitcode layout could differ between translator runs.
    reds_v.sort_by_key(|variable| {
        let kind_rank = match FunctionGraph::concretetype_of(variable) {
            ConcreteType::Signed => 1,
            ConcreteType::GcRef | ConcreteType::Unknown => 2,
            ConcreteType::Float => 3,
            ConcreteType::Void => 4,
        };
        (kind_rank, variable.id())
    });
    reds_v
}

/// Read the rich graph's declared unsigned type for a value while production
/// jtransform still consumes that graph instead of the rtyper-specialized
/// flowspace graph. This is the signedness half of RPython's
/// `Variable.concretetype == lltype.Unsigned`; JIT register kinds intentionally
/// collapse Signed/Unsigned to `'i'`, so `get_value_kind_var` cannot answer it.
fn variable_has_declared_unsigned_type(
    graph: &FunctionGraph,
    variable: &crate::flowspace::model::Variable,
) -> bool {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| {
            if op.result.as_ref() != Some(variable) {
                return false;
            }
            matches!(
                &op.kind,
                OpKind::Input {
                    ty: ValueType::Unsigned,
                    ..
                } | OpKind::FieldRead {
                    ty: ValueType::Unsigned,
                    ..
                } | OpKind::VableFieldRead {
                    ty: ValueType::Unsigned,
                    ..
                } | OpKind::ArrayRead {
                    item_ty: ValueType::Unsigned,
                    ..
                } | OpKind::InteriorFieldRead {
                    item_ty: ValueType::Unsigned,
                    ..
                } | OpKind::VableArrayRead {
                    item_ty: ValueType::Unsigned,
                    ..
                } | OpKind::Call {
                    result_ty: ValueType::Unsigned,
                    ..
                } | OpKind::IndirectCall {
                    result_ty: ValueType::Unsigned,
                    ..
                } | OpKind::BinOp {
                    result_ty: ValueType::Unsigned,
                    ..
                } | OpKind::UnaryOp {
                    result_ty: ValueType::Unsigned,
                    ..
                }
            )
        })
}

impl<'a> Transformer<'a> {
    /// RPython: `Transformer.__init__(cpu=None, callcontrol=None, portal_jd=None)`
    /// (`jtransform.py:62-66`). Pyre keeps `cpu` / `callcontrol` behind
    /// builder setters because the borrow checker demands a late binding
    /// against the enclosing `CallControl`; `portal_jd` follows the same
    /// pattern. All three fields start `None`, matching upstream class
    /// defaults.
    pub fn new(config: &'a GraphTransformConfig) -> Self {
        Self {
            callcontrol: None,
            portal_jd_index: None,
            config,
            vable_array_vars: std::collections::HashMap::new(),
            vable_flags: std::collections::HashMap::new(),
            aliases: std::collections::HashMap::new(),
            notes: Vec::new(),
            vable_rewrites: 0,
            calls_classified: 0,
            analysis_cache: crate::call::AnalysisCache::default(),
            call_argument_vars: Vec::new(),
        }
    }

    /// Set the CallControl for call kind dispatch.
    /// RPython: `Transformer.__init__(callcontrol=...)`.
    pub fn with_callcontrol(mut self, cc: &'a mut crate::call::CallControl) -> Self {
        self.callcontrol = Some(cc);
        self
    }

    /// Attach the portal JitDriverStaticData index for the current
    /// graph. RPython `jtransform.py:65 self.portal_jd = portal_jd`
    /// — "non-None only for the portal graph(s)". Pyre stores the
    /// index into `CallControl::jitdrivers_sd` rather than a direct
    /// reference so the builder does not force a second borrow of
    /// `CallControl`. `handle_jit_marker__jit_merge_point`
    /// (`jtransform.py:1690-1712`) uses this for both identity checks
    /// and `Constant(portal_jd.index, lltype.Signed)` synthesis.
    pub fn with_portal_jd(mut self, jd_index: Option<usize>) -> Self {
        self.portal_jd_index = jd_index;
        self
    }

    /// Accessor for the portal jitdriver index, matching upstream
    /// `self.portal_jd` reads inside Transformer methods.
    pub fn portal_jd_index(&self) -> Option<usize> {
        self.portal_jd_index
    }

    /// RPython: Transformer.transform() — process all blocks in the graph.
    ///
    /// Reads operand kinds via `FunctionGraph::concretetype_of(&v)`
    /// (RPython `getkind(v.concretetype)`).  Callers commit kinds to each
    /// backing `Variable.concretetype` cell upstream — through
    /// `legacy_resolve::resolve_types` (which writes through
    /// `FunctionGraph::set_concretetype_of_inline(&var, ct)` per-set)
    /// or `dual_gate_publish_concretetypes`.  Test fixtures that need
    /// to hand-set kinds call `FunctionGraph::set_concretetype_of_inline(&var, ct)`
    /// directly.
    pub fn transform(&mut self, graph: &FunctionGraph) -> GraphTransformResult {
        let mut rewritten = graph.clone();

        // RPython `rtyper/rpbc.py::SingleFrozenPBCRepr` resolves
        // zero-arg unit-variant PBC ctors to a singleton
        // `Constant(prebuilt_instance_ptr)` before jtransform runs.
        // `transform_graph_to_jitcode` runs this fold on `graph_owned`
        // already; running it again here is idempotent (no-op after
        // the first pass) and ensures `transform_graph` /
        // `transform_graph_with_callcontrol` entry points (test
        // fixtures, etc.) are also covered.
        crate::translator::rtyper::unit_variant_fold::fold_unit_variant_ctors(&mut rewritten);

        // RPython rtyper `specialize_call` rewrites a `we_are_jitted()`
        // `direct_call` to the `_we_are_jitted` symbolic constant
        // (`rpython/rlib/jit.py:403-406`); `rewrite_op_int_is_true` then
        // folds it by symbolic identity (`jtransform.py:1638`).  pyre's
        // rtyper types an ephemeral oracle and never rewrites the
        // surviving model graph, so the symbolic injection runs here —
        // post-annotation, pre-rewrite — keeping the un-annotatable
        // `SpecTag` out of the annotator.
        fold_we_are_jitted_calls(&mut rewritten);

        self.call_argument_vars = rewritten
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|op| match &op.kind {
                OpKind::Call { args, .. } | OpKind::IndirectCall { args, .. } => Some(args),
                _ => None,
            })
            .flatten()
            .cloned()
            .collect();

        let exceptblock = rewritten.exceptblock;
        let graph_name = rewritten.name.clone();
        for block_idx in 0..rewritten.blocks.len() {
            self.optimize_block(&mut rewritten, block_idx, &graph_name, exceptblock);
        }

        GraphTransformResult {
            graph: rewritten,
            notes: std::mem::take(&mut self.notes),
            vable_rewrites: self.vable_rewrites,
            calls_classified: self.calls_classified,
        }
    }

    /// RPython: Transformer.optimize_block()
    fn optimize_block(
        &mut self,
        graph: &mut crate::model::FunctionGraph,
        block_idx: usize,
        graph_name: &str,
        exceptblock: crate::model::BlockId,
    ) {
        // `jtransform.py:74-75 if block.operations == (): return`.  The
        // predicate is the empty *tuple* — `flowspace/model.py:196-197
        // is_final_block` — which only the graph's returnblock and
        // exceptblock carry.  An ordinary block whose operations happen to
        // be empty holds a *list*, and `[] == ()` is false, so it is not
        // skipped.
        if block_idx == graph.returnblock.0 || block_idx == graph.exceptblock.0 {
            return;
        }

        // `jtransform.py:76-77` — both tables are rebound to empty dicts as
        // the first statements of every `optimize_block`, so a virtualizable
        // array read is only visible to the block that performed it.
        self.vable_array_vars.clear();
        self.vable_flags.clear();

        let mut new_ops = Vec::with_capacity(graph.blocks[block_idx].operations.len());

        let original_ops = graph.blocks[block_idx].operations.clone();
        // `jtransform.py:100 count_before_last_operation = len(newoperations)`
        // — recorded on every iteration, so after the loop it holds the count
        // from just before the last operation contributed anything.  `None`
        // is upstream's unbound name: the loop body never ran, so there is no
        // last operation and the elision test below does not apply.
        let mut count_before_last_operation = None;
        for original_op in &original_ops {
            let op = remap_op(original_op, &self.aliases);
            let rewritten = self.rewrite_operation(&op, graph_name, graph);
            count_before_last_operation = Some(new_ops.len());
            match rewritten {
                RewriteResult::Replace(ops) => {
                    new_ops.extend(ops);
                }
                RewriteResult::Identity(alias_target) => {
                    if let Some(result) = op.result.clone() {
                        self.aliases
                            .insert(result, resolve_alias(&alias_target, &self.aliases));
                    }
                }
                RewriteResult::Keep => {
                    new_ops.push(op);
                }
            }
        }

        // `jtransform.py:116-118`: the block's exception exits belong to its
        // last operation.  If rewriting that operation contributed nothing,
        // there is no longer anything here that can raise, so the exception
        // exits and the `last_exception` exitswitch have to go with it —
        // otherwise the block keeps an exception edge for an operation that
        // is not in it any more.
        if graph.blocks[block_idx].canraise() && count_before_last_operation == Some(new_ops.len())
        {
            // `jtransform.py:176-179 _killed_exception_raising_operation`
            let block = &mut graph.blocks[block_idx];
            assert!(
                block.exits[0].exitcase.is_none(),
                "jtransform.py:177 exits[0].exitcase is None"
            );
            block.exits.truncate(1);
            block.exitswitch = None;
        }

        let (exitswitch, exits) = {
            let block = &graph.blocks[block_idx];
            remap_control_flow_metadata_var(
                &block.exitswitch,
                &block.exits,
                |var| remap_value(var, &self.aliases),
                |b| b,
            )
        };
        {
            let block = &mut graph.blocks[block_idx];
            block.operations = new_ops;
            block.exitswitch = exitswitch;
            block.exits = exits;
        }

        // `jtransform.py:123 self.optimize_goto_if_not(block)` — fuse a
        // Bool-producing comparison into the exitswitch
        // (`ExitSwitch::Fused`), eliding the standalone compare op.  Runs
        // after the exitswitch/exits remap is committed; it never alters
        // link targets, so the exceptblock note below is unaffected.
        optimize_goto_if_not(graph, block_idx);

        // `jtransform.py:124-127` — a virtualizable array read must be
        // consumed by the block that performed it: neither a fused
        // exitswitch operand nor a link argument may name one.
        //
        // Upstream interleaves the per-link check with
        // `_do_renaming_on_link` (`jtransform.py:126-128`), so it reads
        // the pre-renaming `link.args`; pyre commits the exits remap in
        // one pass above and therefore reads the post-renaming args.  A
        // tracked array variable is the result of a kept `getfield`, so
        // it is never a renaming *key*; the post-renaming list can only
        // gain occurrences of it, never lose one.
        //
        {
            let block = &graph.blocks[block_idx];
            if let Some(crate::model::ExitSwitch::Fused { args, .. }) = &block.exitswitch {
                self.check_no_vable_array(args.iter(), graph_name, "fused exitswitch operand");
            }
            for link in &block.exits {
                self.check_no_vable_array(
                    link.args.iter().filter_map(LinkArg::as_variable),
                    graph_name,
                    "link argument",
                );
            }
        }

        // Upstream `rpython/translator/backendopt/canraise.py:25-47
        // analyze_exceptblock_in_graph` identifies raising blocks by the
        // presence of a Link in `Block.exits` whose target is
        // `graph.exceptblock`.  pyre records a `GraphTransformNote` for
        // such blocks so later phases (e.g. reporting) can surface
        // unconditional raise sites — the note mirrors the upstream signal
        // without the pyre-specific Terminator::Abort variant.
        let block = &graph.blocks[block_idx];
        if block.exits.iter().any(|link| link.target == exceptblock) {
            self.notes.push(GraphTransformNote {
                function: graph_name.to_string(),
                detail: "abort: raises to exceptblock".to_string(),
            });
        }
    }

    /// `jtransform.py:990-993` — the `fresh_virtualizable` arm of
    /// `is_virtualizable_getset`:
    ///
    /// ```text
    /// if res:
    ///     flags = self.vable_flags[op.args[0]]
    ///     if 'fresh_virtualizable' in flags:
    ///         return False
    /// ```
    ///
    /// A virtualizable that was just allocated is not yet under the
    /// virtualizable protocol — its fields are still ordinary struct
    /// fields — so an access to one of the redirected fields must not
    /// lower to `getfield_vable_*` / `setfield_vable_*`, and an array
    /// field read must not enter `vable_array_vars`.
    ///
    /// Upstream reads `self.vable_flags[op.args[0]]` with a bare
    /// subscript because `rvirtualizable.py:49-53 hook_access_field`
    /// emits a `jit_force_virtualizable` carrying the instance's
    /// annotator flags in front of *every* redirected-field access, so
    /// the entry always exists by the time this runs.  Pyre has no
    /// `hook_access_field`; the flag arrives from the
    /// `hint(x, fresh_virtualizable=True)` call itself
    /// (`rlib/jit.py:321-322`), so an absent entry is the ordinary
    /// "not fresh" case rather than a missing preceding op.
    ///
    /// It can also mean "hinted, but not in this block": the hint files
    /// one entry in the block it sits in, and `vable_flags` is rebuilt
    /// per block.  See `rewrite_op_hint`'s `FreshVirtualizable` arm for
    /// why upstream does not have that constraint.
    ///
    /// No pyre source calls the hint today, so this arm is not the one
    /// that fires: the reachable path is the structural sibling below,
    /// `FieldDescriptor::base_is_local_aggregate`, which recovers the
    /// same fact from the access reaching a container that was never
    /// dereferenced.  Keep this arm anyway — the two are not the same
    /// claim, and the structural one is not a superset.  "The container
    /// is a value in this frame" and "the programmer asserts this object
    /// is freshly allocated" diverge for a frame that is heap-allocated
    /// and then initialised *through* the pointer: fresh, but reached by
    /// a deref, so only the hint can see it.  The frame constructor takes
    /// its frame by value today, which is exactly why the structural arm
    /// suffices; this arm is what survives that becoming by-pointer.
    ///
    /// That by-value parameter is not a JIT-imposed constraint, so do not
    /// "fix" it on the JIT's behalf: the constructor publishes every GCREF
    /// field into the root bracket, allocates, reads them back, then does a
    /// single `ptr::write` of the whole aggregate.  Allocating first would
    /// leave those fields in a heap object live across the allocation —
    /// the situation the `push_roots` / `pin_roots` / `get` dance exists to
    /// avoid for a stack aggregate — and would cost a write barrier per
    /// field instead of one at the end.  The GC and the JIT want the same
    /// shape here; they simply want it for different reasons.
    fn is_fresh_virtualizable(&self, base: &crate::flowspace::model::Variable) -> bool {
        self.vable_flags.get(base) == Some(&VableFlag::FreshVirtualizable)
    }

    /// `jtransform.py:145-168 _check_no_vable_array`.
    ///
    /// `vable_array_vars` is rebuilt per block, so the array a
    /// virtualizable field read produced can only be lowered by an
    /// array access sitting in that same block.  Any other consumer —
    /// a link argument, a fused exitswitch operand, a call argument, a
    /// `setfield` — would carry the raw array pointer out of reach of
    /// the lowering, so the graph is rejected instead.
    ///
    /// # What the check needed before it could be wired
    ///
    /// The four call sites are the fused-exitswitch and link-args pair in
    /// `optimize_block` (`jtransform.py:124-127`),
    /// `rewrite_call_three_lists` (`jtransform.py:421`), and
    /// `rewrite_op_setfield` (`jtransform.py:912`).  All four stood
    /// commented out for a while, because the check was correct and what
    /// it rejected was not.
    ///
    /// pyre's MIR front used to give a block its first predecessor's exit
    /// framestate verbatim, so its inputargs were the *same* `Variable`
    /// identities as the link args feeding them — measured at 142598 of
    /// 153927 blocks.  That defeats `transform_dead_op_vars`' positional
    /// liveness (`simplify.py:512-524`): a value read anywhere in the
    /// predecessor is in `read_vars`, so the dead link arg naming it is
    /// never dropped, and the array leaves its block on a link upstream
    /// would have pruned away.
    ///
    /// The witnessed case was `pyre_interpreter::pyframe::<Impl>::_check_stack_index`
    /// (`index >= self.stack_base() && index < locals_w!(self).len()`).
    /// Its array read and the `ArrayLen` consuming it sit in the same
    /// block, exactly as the protocol requires — and the array still rode
    /// that block's outgoing link **twice**, into a block whose only
    /// operation is the `int_lt` and which never touches it.  Eliminating
    /// empty blocks does not reach that: the block it escapes into is not
    /// empty.
    ///
    /// Two things fixed it, and each is load-bearing on its own —
    /// reverting either one puts the array back on the link.
    /// `lower_framestate` now copies the entry framestate
    /// (`flowcontext.py:466 newstate = state.copy()`), and
    /// `simplify_lowered_graph` runs `prune_dead_phis` unconditionally, as
    /// `all_passes` does (simplify.py:1067).
    ///
    /// `tests/test_vable_array_len.rs` holds the regression: not just
    /// "the `len` lowered to `ArrayLen`", but that the array appears in no
    /// link's args and is no block's inputarg anywhere in the graph.
    /// Enumerating individual escape routes is what let `_check_stack_index`
    /// pass a green test while escaping by the one route that test did not
    /// look at.
    ///
    /// `route` names which of the four the escape took.  Upstream's
    /// message does not carry it, and the extra line is a deliberate pyre
    /// addition: the four call sites below share one panic body, this
    /// function is small enough to be inlined into all of them in an
    /// optimised build, and the resulting backtrace then names whichever
    /// site the optimiser happened to keep.  A misread frame sent a whole
    /// diagnosis round after a residual call for what was a link argument.
    /// Keep the upstream block verbatim above and add the route last.
    ///
    /// The four routes are closed — the call sites are the only ones —
    /// and each passes its own literal: `"link argument"`,
    /// `"fused exitswitch operand"`, `"call argument"`,
    /// `"setfield operand"` (`jtransform.py:912`, the base or the stored
    /// value of a `setfield`; there is no `setarrayitem` site, because
    /// that one is the array access the protocol exists to allow).
    fn check_no_vable_array<'v>(
        &self,
        list: impl IntoIterator<Item = &'v crate::flowspace::model::Variable>,
        graph_name: &str,
        route: &str,
    ) {
        // `jtransform.py:146-147`
        if self.vable_array_vars.is_empty() {
            return;
        }
        for v in list {
            if let Some((_v_base, array_index, _itemsize, _is_signed)) =
                self.vable_array_vars.get(v)
            {
                panic!(
                    "A virtualizable array is passed around; it should\n\
                     only be used immediately after being read.  Note\n\
                     that a possible cause is indexing with an index not\n\
                     known non-negative, or catching IndexError, or\n\
                     not inlining at all (for tests: use listops=True).\n\
                     This is about: vable array field #{array_index}\n\
                     Occurred in: {graph_name}\n\
                     Escaped via: {route}"
                );
            }
            // extra explanation: with the way things are organized in
            // rpython/rlist.py, the ll_getitem becomes a function call
            // that is typically meant to be inlined by the JIT, but
            // this does not work with vable arrays because
            // jtransform.py expects the getfield and the getarrayitem
            // to be in the same basic block.  It works a bit as a hack
            // for simple cases where we performed the backendopt
            // inlining before (even with a very low threshold, because
            // there is _always_inline_ on the relevant functions).
        }
    }

    /// `jtransform.py:421-422` — the two opening statements of
    /// `rewrite_call`: reject a virtualizable array among the call's
    /// arguments, then split the surviving arguments by kind.  pyre has
    /// no single `rewrite_call`; each per-call-kind handler reaches this
    /// pair directly, so the pairing lives here.
    fn rewrite_call_three_lists(
        &self,
        args: &[crate::flowspace::model::Variable],
        graph_name: &str,
    ) -> (
        Vec<crate::flowspace::model::Variable>,
        Vec<crate::flowspace::model::Variable>,
        Vec<crate::flowspace::model::Variable>,
    ) {
        self.check_no_vable_array(args.iter(), graph_name, "call argument");
        self.make_three_lists_from_vars(args)
    }

    /// Variable-returning helper —
    /// allocates a fresh synthetic slot, stamps its `concretetype` cell
    /// to `ty`, and hands back the backing
    /// [`crate::flowspace::model::Variable`].  Call sites that hold the
    /// fresh slot only as a Variable handle (`op.result.clone()`
    /// downstream) use this to skip the local `must_variable`
    /// projection.  RPython parity:
    /// `Variable(concretetype=T)` (`flowspace/model.py:Variable.__init__`).
    fn fresh_synthetic_variable_typed(
        &mut self,
        graph: &mut FunctionGraph,
        ty: crate::codewriter::type_state::ConcreteType,
    ) -> crate::flowspace::model::Variable {
        graph.alloc_value_var_with_type(ty)
    }

    /// RPython parity: `Variable.concretetype = ty` (`flowmodel.py
    /// Variable.__init__`).  Updates the backing Variable's
    /// `concretetype` cell in-place; the optional shape lets the
    /// jtransform rewrite arms call this with `op.result.clone()`
    /// for result-less ops without an extra guard.
    fn stamp_value_kind(
        &mut self,
        _graph: &FunctionGraph,
        value: Option<crate::flowspace::model::Variable>,
        ty: crate::codewriter::type_state::ConcreteType,
    ) {
        if let Some(var) = value {
            FunctionGraph::set_concretetype_of_inline(&var, ty);
        }
    }

    /// Stamp the synthetic kind from a `ValueType` source, skipping
    /// `ValueType::Unknown` so the fallback Unknown→Ref defaulting in
    /// `value_type_to_kind` does not clobber a real `concretetype`
    /// already computed by the rtyper for an existing Variable.
    fn stamp_value_kind_from_value_type(
        &mut self,
        graph: &FunctionGraph,
        value: Option<crate::flowspace::model::Variable>,
        result_ty: &ValueType,
    ) {
        let ty = match result_ty {
            ValueType::Int | ValueType::Unsigned | ValueType::Bool | ValueType::State => {
                crate::codewriter::type_state::ConcreteType::Signed
            }
            ValueType::Ref(_) | ValueType::Str => {
                crate::codewriter::type_state::ConcreteType::GcRef
            }
            ValueType::Float => crate::codewriter::type_state::ConcreteType::Float,
            ValueType::Void => crate::codewriter::type_state::ConcreteType::Void,
            ValueType::Unknown | ValueType::Int128 | ValueType::UInt128 => return,
        };
        self.stamp_value_kind(graph, value, ty);
    }

    /// RPython `op.result.concretetype` is set by the rtyper, so jtransform
    /// reads it directly. Pyre's front-end can leave a callee return type
    /// as `ValueType::Unknown` when the callee path is unresolved (re-export
    /// shadowing, cross-crate paths). When that happens, the rtyper's
    /// backward-inference pass classifies `op.result` from its consumer-op
    /// constraints, and that classification is stamped on the result
    /// Variable's `concretetype` cell via `apply_to_graph` before this
    /// pass runs.  Reading `FunctionGraph::concretetype_of(&v)` therefore
    /// propagates the same `result_kind` the rtyper already chose,
    /// instead of
    /// falling back to `value_type_to_kind(Unknown) == 'r'`.
    fn resolve_call_result_kind(
        &self,
        result: Option<&crate::flowspace::model::Variable>,
        result_ty: &ValueType,
    ) -> Option<char> {
        if !matches!(result_ty, ValueType::Unknown) {
            return None;
        }
        let var = result?;
        match FunctionGraph::concretetype_of(var) {
            crate::codewriter::type_state::ConcreteType::Signed => Some('i'),
            crate::codewriter::type_state::ConcreteType::GcRef => Some('r'),
            crate::codewriter::type_state::ConcreteType::Float => Some('f'),
            crate::codewriter::type_state::ConcreteType::Void => Some('v'),
            crate::codewriter::type_state::ConcreteType::Unknown => None,
        }
    }

    /// RPython uses one `op.result.concretetype` for both
    /// `getkind(...)[0]` (opcode suffix) and `getcalldescr(..., RESULT)`.
    /// Keep the Rust port on that same source of truth so `Unknown`
    /// results resolved by `type_state` cannot produce `_i` opnames with
    /// Ref-return calldescrs.
    fn resolve_call_result(
        &self,
        result: Option<&crate::flowspace::model::Variable>,
        result_ty: &ValueType,
    ) -> ResolvedCallResult {
        // A result-less call op is Void at the IR level even when the
        // callee declares a scalar return (the front drops unused
        // results; RPython would carry a Void-typed result Variable
        // here, and `getkind(Void)` selects the `_v` opname).  Without
        // this the assembler emits e.g. `residual_call_r_i/iRd` — an
        // `_i` opname with no `>i` result slot, a key no blackhole
        // handler has.
        let kind = if result.is_none() {
            'v'
        } else {
            self.resolve_call_result_kind(result, result_ty)
                .unwrap_or_else(|| value_type_to_kind(result_ty))
        };
        let ir_type = match kind {
            'i' => majit_ir::value::Type::Int,
            'r' => majit_ir::value::Type::Ref,
            'f' => majit_ir::value::Type::Float,
            'v' => majit_ir::value::Type::Void,
            other => panic!("unsupported call result kind '{other}'"),
        };
        ResolvedCallResult { kind, ir_type }
    }

    /// Reconcile a call's front-derived result type with the callee's
    /// post-`?` declared `RESULT` (`call.py:230` `RESULT == FUNC.RESULT`).
    ///
    /// A `Result<(), PyError>` scoped callee declares `FUNC.RESULT = void`,
    /// but `front::mir` types the call's result `Ref`: every aggregate,
    /// including the unit `()`, lowers to `Ref`.  This is the caller-side
    /// mirror of the `declared=='v' && cfg=='r'` returnblock widen
    /// `finalize_rewritten_graph_to_jitcode` runs on the callee — reconcile
    /// the call to the declared void so the residual_call emits a `_v`
    /// opname with no result slot.  Emitting the `Ref` slot would hand GC a
    /// register holding the void call's garbage.  Only the
    /// declared-void-vs-resolved-Ref shell case is rewritten; every other
    /// call keeps its front-derived `result_ty`.
    fn effective_call_result_ty(
        &self,
        target: &CallTarget,
        op_result: Option<&crate::flowspace::model::Variable>,
        result_ty: &ValueType,
    ) -> ValueType {
        let declared_void = self
            .callcontrol
            .as_deref()
            .and_then(|cc| cc.declared_result_type_for_target(target))
            == Some(majit_ir::value::Type::Void);
        if declared_void && self.resolve_call_result(op_result, result_ty).kind == 'r' {
            ValueType::Void
        } else {
            result_ty.clone()
        }
    }

    fn direct_funcptr_value(
        &mut self,
        graph: &mut FunctionGraph,
        target: &CallTarget,
    ) -> (crate::flowspace::model::Variable, SpaceOperation) {
        let fnaddr = self
            .callcontrol
            .as_deref()
            .map(|cc| cc.fnaddr_for_target(target))
            .unwrap_or_else(|| crate::call::symbolic_fnaddr_for_target(target));
        // Function pointer materialized as ConstInt — assembler emits it
        // through the `'i'` argcode so the kind is Signed.
        let var = self.fresh_synthetic_variable_typed(
            graph,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        (
            var.clone(),
            SpaceOperation {
                result: Some(var),
                kind: OpKind::ConstInt(fnaddr),
            },
        )
    }

    fn materialize_fn_const_funcptr_value(
        &mut self,
        graph: &mut FunctionGraph,
        funcptr: &crate::flowspace::model::Variable,
    ) -> Option<(crate::flowspace::model::Variable, SpaceOperation)> {
        let target = fn_const_target_for_var(graph, funcptr, 0)?;
        Some(self.direct_funcptr_value(graph, &target))
    }

    /// RPython: Transformer.rewrite_operation() — dispatch to rewrite_op_*.
    fn rewrite_operation(
        &mut self,
        op: &SpaceOperation,
        graph_name: &str,
        graph: &mut crate::model::FunctionGraph,
    ) -> RewriteResult {
        match &op.kind {
            // ── rewrite_op_hint ──
            //
            // The structured `OpKind::Hint` (emitted by `front::mir` for
            // `jit::promote` / `#[elidable_promote]`) carries the hint kind
            // directly.  A bare `Call` to a hint-marker path is still
            // accepted (defensive / unit tests construct it) and classified
            // by name.
            OpKind::Hint { value, kind } => {
                self.rewrite_op_hint(op, *kind, std::slice::from_ref(value), "hint", graph_name)
            }
            OpKind::Call { target, args, .. } if classify_hint_target(target).is_some() => {
                let kind = classify_hint_target(target).expect("guard checked Some");
                let label = target.to_string();
                self.rewrite_op_hint(op, kind, args, &label, graph_name)
            }
            // ── fold of the `_we_are_jitted` symbolic ──
            //
            // Inside the tracer / blackhole interpreter `we_are_jitted()`
            // is always true (`rlib/jit.py:355`).  RPython folds this at
            // `rewrite_op_int_is_true` (`jtransform.py:1638`) keyed on the
            // symbolic value identity (`value is _we_are_jitted`).  pyre's
            // `we_are_jitted() -> bool` carries the symbolic as
            // `OpKind::ConstSymbolic { tag, .. }` (injected by
            // `front::mir`); fold it to `ConstBool(true)` keyed on the
            // `SpecTag` identity — the dual of `constfold::
            // replace_we_are_jitted` folding the genc side to `false`.
            OpKind::ConstSymbolic { tag, .. }
                if *tag == crate::translator::backendopt::constfold::WE_ARE_JITTED_TAG_ID =>
            {
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::ConstBool(true),
                }])
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
                self.rewrite_op_setfield(op, field, value, ty, graph_name)
            }
            // ── rewrite_op_getarrayitem ──
            OpKind::ArrayRead {
                base,
                index,
                item_ty,
                ..
            } if self.config.lower_virtualizable => {
                self.rewrite_op_getarrayitem(op, base, index, item_ty, graph_name)
            }
            // ── rewrite_op_setarrayitem ──
            OpKind::ArrayWrite {
                base,
                index,
                value,
                item_ty,
                ..
            } if self.config.lower_virtualizable => {
                self.rewrite_op_setarrayitem(op, base, index, value, item_ty, graph_name)
            }
            // ── rewrite_op_direct_call ──
            OpKind::Call {
                target,
                args,
                result_ty,
            } if self.config.classify_calls => {
                self.rewrite_op_direct_call(op, target, args, result_ty, graph_name, graph)
            }
            // ── rewrite_op_indirect_call ──
            // RPython jtransform.py:410-412. Pyre's rtyper-equivalent
            // (`translator/rtyper/rpbc.rs`) lowers
            // `OpKind::Call { target: CallTarget::Indirect, .. }` into
            // `OpKind::IndirectCall { funcptr, args, graphs, .. }`
            // before jtransform runs, so by this point the funcptr is
            // already a regular Variable and `args` still carry the full
            // call argument list, including the receiver.
            OpKind::IndirectCall {
                funcptr,
                args,
                graphs,
                result_ty,
            } if self.config.classify_calls => self.lower_indirect_call_op(
                op,
                funcptr,
                args,
                graphs.as_deref(),
                result_ty,
                graph_name,
                graph,
            ),
            // ── abort placeholders ──
            OpKind::Abort { kind } => {
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!("abort placeholder: {:?}", kind),
                });
                RewriteResult::Keep
            }
            // `jtransform.py:384-392` `rewrite_op_int_add_ovf` (aliased to
            // `rewrite_op_int_mul_ovf`) and `rewrite_op_int_sub_ovf` both
            // return `[SpaceOperation('-live-', [], None), op]`.  `flatten`
            // pops the overflow-checked op back off the emitted line and
            // re-spells it as `int_*_jump_if_ovf`, a guard that resumes at
            // this point — the `-live-` in front is what makes that a valid
            // resume point rather than one that merely happens to inherit a
            // marker from the preceding operation.
            OpKind::BinOp { op: binop_name, .. }
                if matches!(binop_name.as_str(), "add_ovf" | "sub_ovf" | "mul_ovf") =>
            {
                RewriteResult::Replace(vec![
                    SpaceOperation {
                        result: None,
                        kind: OpKind::Live,
                    },
                    op.clone(),
                ])
            }
            OpKind::BinOp {
                op: binop_name,
                lhs,
                rhs,
                result_ty,
            } if matches!(binop_name.as_str(), "bitand" | "bitor" | "bitxor") => {
                let canonical = match binop_name.as_str() {
                    "bitand" => "and",
                    "bitor" => "or",
                    "bitxor" => "xor",
                    _ => unreachable!("matched bit op names above"),
                };
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::BinOp {
                        op: canonical.into(),
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                        result_ty: result_ty.clone(),
                    },
                }])
            }
            OpKind::UnaryOp {
                op: unop_name,
                operand,
                ..
            } if unop_name == "same_as"
                || unop_name == "deref"
                || unop_name == "cast_bool_to_int"
                || unop_name == "cast_bool_to_uint"
                || unop_name == "cast_int_to_uint"
                || unop_name == "cast_uint_to_int" =>
            {
                // RPython `jtransform.py:246-248 rewrite_op_same_as`:
                //
                //     def rewrite_op_same_as(self, op):
                //         if op.args[0] in self.vable_array_vars:
                //             self.vable_array_vars[op.result] = \
                //                 self.vable_array_vars[op.args[0]]
                //
                // `rewrite_op_same_as` returns `None` implicitly.  In
                // `optimize_block` that means "remove the op and rename
                // the result to args[0]" (`jtransform.py:106-111`).
                //
                // `cast_bool_to_int` / `cast_bool_to_uint` /
                // `cast_int_to_uint` / `cast_uint_to_int` follow the
                // same drop-and-alias shape — RPython
                // `jtransform.py:330,331,336,337`:
                //
                //     def rewrite_op_cast_bool_to_int(self, op): pass
                //     def rewrite_op_cast_bool_to_uint(self, op): pass
                //     def rewrite_op_cast_int_to_uint(self, op): pass
                //     def rewrite_op_cast_uint_to_int(self, op): pass
                //
                // are explicit no-ops.  Each cast is identity at LL
                // level because `getkind(lltype.Bool) == getkind(
                // lltype.Signed) == getkind(lltype.Unsigned) == 'int'`
                // (`history.py:45-63`), so backend opcodes need not
                // grow separate handlers.
                //
                // Pyre instead drops the op (`RewriteResult::Identity`)
                // and records `op.result -> *operand` in `self.aliases`
                // (line 446-450). Subsequent ops in the same block
                // (and `block.exitswitch` / `link.args` via
                // `remap_control_flow_metadata`) go through `remap_op`
                // at line 441 before dispatch, so a consumer that
                // originally referenced `op.result` is rewritten to
                // reference `*operand` directly. The later
                // `vable_array_vars.get(&base)` lookup in
                // `rewrite_op_getarrayitem`/`_setarrayitem` then hits
                // the original `(*operand)` entry — same outcome as
                // upstream's explicit propagation, without keeping a
                // redundant alias key.
                RewriteResult::Identity(operand.clone())
            }
            // `ll_str` on a string is identity (`rpython/rtyper/
            // lltypesystem/rstr.py` `ll_str` returns the string
            // unchanged).  The front-end's `format!` / `to_string`
            // expansion emits `str(x)` for a string-typed value; over a
            // Ref (string) operand it folds to a no-op alias so it does
            // not fall through to the unwired `int_str/r>r` default.  A
            // `str` over an Int operand keeps the integer render path.
            OpKind::UnaryOp {
                op: unop_name,
                operand,
                ..
            } if unop_name == "str" && self.get_value_kind_var(operand) == 'r' => {
                RewriteResult::Identity(operand.clone())
            }
            // `str` over an unboxed integer renders the decimal string.
            // `rint.py rtype_str` / `rstr.py ll_int2dec` lower `str(int)`
            // to a `direct_call` of the render helper during rtyping, so
            // the blackhole never sees a bare `int_str` op.  Pyre keeps
            // `str(x)` as `UnaryOp { op: "str" }` through the graph; over
            // an Int operand it lowers to the registered `jit_int_str`
            // host extern (`pyre_object::unicodeobject`, address in
            // `jit_fnaddr.rs`).  An int-only residual selects the
            // canonical `ir` shape with an empty ref list
            // (`assembler.rs` `emit_canonical_call_void` `has_int` arm),
            // so it assembles to the wired `residual_call_ir_r/iIRd>r`.
            // Without this the op falls through to the unwired
            // `int_str/i>r` default.  Like `_ll_2_int_*` (route-(a)) the
            // helper carries no oopspec; `ElidableCanRaise` mirrors the
            // sibling `jit_str_concat` allocating string helper.
            OpKind::UnaryOp {
                op: unop_name,
                operand,
                ..
            } if unop_name == "str" && self.get_value_kind_var(operand) == 'i' => {
                let target = CallTarget::function_path(["jit_int_str"]);
                let (funcptr, funcptr_op) = self.direct_funcptr_value(graph, &target);
                let mut ops = vec![funcptr_op];
                ops.push(SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::CallResidual {
                        funcptr: CallFuncPtr::Value(funcptr),
                        descriptor: CallDescriptor::from_signature(
                            &[majit_ir::value::Type::Int],
                            majit_ir::value::Type::Ref,
                            EffectInfo::new(ExtraEffect::ElidableCanRaise, OopSpecIndex::None),
                        ),
                        args_i: vec![operand.clone()],
                        args_r: vec![],
                        args_f: vec![],
                        result_kind: 'r',
                        indirect_targets: None,
                    },
                });
                ops.push(SpaceOperation {
                    result: None,
                    kind: OpKind::Live,
                });
                RewriteResult::Replace(ops)
            }
            // RPython `jtransform.py:1592` rename pass:
            //   ('cast_bool_to_float', 'cast_int_to_float'),
            // The Bool register class is 'int' at LL, so the same
            // `cast_int_to_float` machine op handles the conversion;
            // backend opcodes don't need a separate `cast_bool_to_float`.
            OpKind::UnaryOp {
                op: unop_name,
                operand,
                ..
            } if unop_name == "cast_bool_to_float" => {
                self.stamp_value_kind(
                    graph,
                    op.result.clone(),
                    crate::codewriter::type_state::ConcreteType::Float,
                );
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::UnaryOp {
                        op: "cast_int_to_float".into(),
                        operand: operand.clone(),
                        result_ty: ValueType::Float,
                    },
                }])
            }
            // RPython `jtransform.py:1243-1255` `rewrite_op_ptr_eq`/`rewrite_op_ptr_ne`
            // + `_rewrite_cmp_ptrs`: equality/inequality of two Ref operands is
            // `ptr_eq`/`ptr_ne` (wired at `blackhole.py:585-590`), not `int_eq`/
            // `int_ne`. Pyre's front-end emits a unified `BinOp { op: "eq"/"ne" }`
            // because Rust's `==`/`!=` is one AST node regardless of operand type;
            // the jtransform layer is where RPython branches on operand kind.
            // Both operands Ref → rewrite to `ptr_eq`/`ptr_ne`. Mixed/Int operands
            // stay as `int_eq`/`int_ne`.
            OpKind::BinOp {
                op: binop_name,
                lhs,
                rhs,
                result_ty,
            } if (binop_name == "eq" || binop_name == "ne")
                && self.get_value_kind_var(lhs) == 'r'
                && self.get_value_kind_var(rhs) == 'r' =>
            {
                let new_op = if binop_name == "eq" {
                    "ptr_eq"
                } else {
                    "ptr_ne"
                };
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::BinOp {
                        op: new_op.into(),
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                        result_ty: result_ty.clone(),
                    },
                }])
            }
            // jtransform.py:1227-1235 `_rewrite_cmp_ptrs`: non-GC ptr
            // equality becomes int_eq/int_ne.  Mixed i+r operands from
            // Pyre's unified BinOp need cast_ptr_to_int on the Ref side
            // (blackhole.py:603-606 `cast_ptr_to_int/r>i` is wired).
            OpKind::BinOp {
                op: binop_name,
                lhs,
                rhs,
                result_ty,
            } if (binop_name == "eq" || binop_name == "ne")
                && (self.get_value_kind_var(lhs) == 'r' || self.get_value_kind_var(rhs) == 'r') =>
            {
                let (lhs_coerced, mut ops) = self.coerce_operand_to_int(graph, lhs);
                let (rhs_coerced, rhs_ops) = self.coerce_operand_to_int(graph, rhs);
                ops.extend(rhs_ops);
                ops.push(SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::BinOp {
                        op: binop_name.clone(),
                        lhs: lhs_coerced,
                        rhs: rhs_coerced,
                        result_ty: result_ty.clone(),
                    },
                });
                RewriteResult::Replace(ops)
            }
            // RPython rtyper produces `float_*` opnames directly when
            // operand `concretetype` is `lltype.Float` — there is no
            // `int_*` op with float operands in RPython's IR
            // (`rpython/rtyper/rfloat.py` rtype_add etc. emit
            // `float_add` / `float_sub` / `float_mul` / `float_truediv`
            // / `float_lt` / `float_eq` etc.).  Pyre's front-end emits
            // a unified `OpKind::BinOp { op: "add" }` because Rust's
            // `+` is one AST node regardless of operand type; this
            // jtransform pass mirrors RPython's rtyper-level
            // distinction by rewriting `int_*` over Float operands to
            // `float_*`.  Both operands Float → arithmetic returns
            // Float; comparisons return Int (bool).  RPython's rtyper
            // inserts `cast_int_to_float` for mixed int/float pairs
            // before emitting the `float_*` op; pyre's lighter rtyper
            // leaves the generic BinOp in place, so jtransform performs
            // that local coercion here.
            //
            // `mod` is handled separately: RPython does not provide
            // `float_mod` (`rpython/rtyper/lltypesystem/lloperation.py:260`
            // "don't implement float_mod, use math.fmod instead"), so
            // `%` over floats lowers to a residual `ll_math_fmod` call.
            OpKind::BinOp {
                op: binop_name,
                lhs,
                rhs,
                result_ty,
            } if canonical_float_arith_binop(binop_name).is_some()
                && is_float_rewrite_domain(self.get_value_kind_var(lhs))
                && is_float_rewrite_domain(self.get_value_kind_var(rhs))
                && (self.get_value_kind_var(lhs) == 'f'
                    || self.get_value_kind_var(rhs) == 'f'
                    || *result_ty == ValueType::Float) =>
            {
                let canonical = match canonical_float_arith_binop(binop_name)
                    .expect("guard checked float arithmetic op")
                {
                    "div" => "truediv", // RPython lltype: `float_truediv`
                    other => other,
                };
                self.stamp_value_kind(
                    graph,
                    op.result.clone(),
                    crate::codewriter::type_state::ConcreteType::Float,
                );
                let (lhs, mut ops) = self.coerce_operand_to_float_domain(graph, lhs);
                let (rhs, rhs_ops) = self.coerce_operand_to_float_domain(graph, rhs);
                ops.extend(rhs_ops);
                ops.push(SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::BinOp {
                        op: format!("float_{canonical}"),
                        lhs,
                        rhs,
                        result_ty: ValueType::Float,
                    },
                });
                RewriteResult::Replace(ops)
            }
            OpKind::BinOp {
                op: binop_name,
                lhs,
                rhs,
                ..
            } if matches!(binop_name.as_str(), "lt" | "le" | "gt" | "ge" | "eq" | "ne")
                && is_float_rewrite_domain(self.get_value_kind_var(lhs))
                && is_float_rewrite_domain(self.get_value_kind_var(rhs))
                && (self.get_value_kind_var(lhs) == 'f' || self.get_value_kind_var(rhs) == 'f') =>
            {
                self.stamp_value_kind(
                    graph,
                    op.result.clone(),
                    crate::codewriter::type_state::ConcreteType::Signed,
                );
                let (lhs, mut ops) = self.coerce_operand_to_float_domain(graph, lhs);
                let (rhs, rhs_ops) = self.coerce_operand_to_float_domain(graph, rhs);
                ops.extend(rhs_ops);
                ops.push(SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::BinOp {
                        op: format!("float_{binop_name}"),
                        lhs,
                        rhs,
                        result_ty: ValueType::Int,
                    },
                });
                RewriteResult::Replace(ops)
            }
            // PRE-EXISTING-ADAPTATION (no direct RPython precedent): pyre-
            // side recovery when integer comparisons reach jtransform with
            // a Ref-typed operand because the rtyper-equivalent did not
            // stamp the operand's `concretetype` (or an `lltype.
            // cast_ptr_to_int` was elided from the SSA chain).  RPython's
            // rtyper inserts the cast at the rtyper layer
            // (`rpython/rtyper/rint.py`), so by the time `jtransform.py`
            // observes the comparison the operands are uniformly `Signed`;
            // pyre's lighter rtyper leaves the generic `BinOp` in place
            // with one or both operands defaulting to `'r'` kind, so the
            // unconditional `int_<op>` prefix at
            // `assembler.rs:3160` would emit `int_eq/ir>i` /
            // `int_le/ri>i` opnames that no RPython blackhole handler
            // registers (see
            // `default_bh_builder_unwired_set_matches_task_85_snapshot`).
            //
            // Coverage covers all six
            // comparison ops (`eq`/`ne`/`lt`/`le`/`gt`/`ge`).  The
            // earlier "eq/ne only" restriction surfaced `int_le/r*`
            // as unwired blackhole opnames, breaking the expected-empty
            // snapshot.  RPython has no `ptr_lt` family,
            // but `cast_ptr_to_int` followed by `int_lt`/`int_le`
            // matches what `rpython/rtyper/rint.py` emits for any
            // comparison whose operands cross the ptr/int boundary —
            // the cast is rtyper-orthodox, the resulting `int_<cmp>/ii>i`
            // opname is wired by the blackhole.  Producer-side fix
            // for the missing rtyper cast remains the canonical
            // convergence path; this jtransform recovery is the
            // bridge until that lands.
            // eq/ne with BOTH operands ref-kind → emit ptr_eq / ptr_ne
            // directly.  PyPy `rpython/rtyper/rptr.py:167-184
            // pairtype(PtrRepr, Repr).rtype_eq/ne` calls
            // `hop.inputargs(r_ptr, r_ptr)` (both already ptr-typed in
            // this branch — no cast) and emits `ptr_eq` / `ptr_ne`.
            // Pyre's blackhole has `bhimpl_ptr_eq` / `bhimpl_ptr_ne`
            // wired at `bh_binop_r_to_i`, so the resulting
            // `ptr_eq/rr>i` opname dispatches without going through
            // `cast_ptr_to_int`.
            OpKind::BinOp {
                op: binop_name,
                lhs,
                rhs,
                result_ty,
            } if matches!(binop_name.as_str(), "eq" | "ne")
                && self.get_value_kind_var(lhs) == 'r'
                && self.get_value_kind_var(rhs) == 'r' =>
            {
                self.stamp_value_kind(
                    graph,
                    op.result.clone(),
                    crate::codewriter::type_state::ConcreteType::Signed,
                );
                let ptr_op = if binop_name == "eq" {
                    "ptr_eq"
                } else {
                    "ptr_ne"
                };
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::BinOp {
                        op: ptr_op.into(),
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                        result_ty: result_ty.clone(),
                    },
                }])
            }
            // Mixed-kind eq/ne (one ref + one int) or any ordered
            // ref-cmp (lt/le/gt/ge with a ref operand) — PRE-EXISTING
            // ADAPTATION: pyre's frontend admits source patterns
            // RPython does not (PyPy `rptr.py` only registers eq/ne for
            // PtrRepr pairtype, never `<`/`<=`/`>`/`>=`; mixed
            // ref+int eq/ne would surface as a TyperError at PyPy's
            // `inputargs(r_ptr, r_ptr)` convertfromrepr step).  Pyre
            // bridges by coercing every ref operand through
            // `cast_ptr_to_int` and emitting `int_<op>`.  The canonical
            // PyPy-orthodox close is fixing the source patterns
            // upstream (use `is_null()` / explicit `as` cast) or
            // moving the cast emission into pyre's rtyper rint
            // compare-template; this is not yet implemented.
            OpKind::BinOp {
                op: binop_name,
                lhs,
                rhs,
                result_ty,
            } if matches!(binop_name.as_str(), "eq" | "ne" | "lt" | "le" | "gt" | "ge")
                && matches!(self.get_value_kind_var(lhs), 'i' | 'r')
                && matches!(self.get_value_kind_var(rhs), 'i' | 'r')
                && (self.get_value_kind_var(lhs) == 'r' || self.get_value_kind_var(rhs) == 'r') =>
            {
                self.stamp_value_kind(
                    graph,
                    op.result.clone(),
                    crate::codewriter::type_state::ConcreteType::Signed,
                );
                let (lhs_var, lhs_pre_ops) = self.coerce_operand_to_int(graph, lhs);
                let (rhs_var, rhs_pre_ops) = self.coerce_operand_to_int(graph, rhs);
                let mut ops = lhs_pre_ops;
                ops.extend(rhs_pre_ops);
                ops.push(SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::BinOp {
                        op: binop_name.clone(),
                        lhs: lhs_var,
                        rhs: rhs_var,
                        result_ty: result_ty.clone(),
                    },
                });
                RewriteResult::Replace(ops)
            }
            OpKind::BinOp {
                op: binop_name,
                lhs,
                rhs,
                result_ty,
            } if canonical_float_mod_binop(binop_name).is_some()
                && is_float_rewrite_domain(self.get_value_kind_var(lhs))
                && is_float_rewrite_domain(self.get_value_kind_var(rhs))
                && (self.get_value_kind_var(lhs) == 'f'
                    || self.get_value_kind_var(rhs) == 'f'
                    || *result_ty == ValueType::Float) =>
            {
                self.stamp_value_kind(
                    graph,
                    op.result.clone(),
                    crate::codewriter::type_state::ConcreteType::Float,
                );
                let (lhs, mut ops) = self.coerce_operand_to_float_domain(graph, lhs);
                let (rhs, rhs_ops) = self.coerce_operand_to_float_domain(graph, rhs);
                ops.extend(rhs_ops);
                let target = CallTarget::function_path(["ll_math_fmod"]);
                let (funcptr, funcptr_op) = self.direct_funcptr_value(graph, &target);
                ops.push(funcptr_op);
                ops.push(SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::CallResidual {
                        funcptr: CallFuncPtr::Value(funcptr),
                        descriptor: CallDescriptor::from_signature(
                            &[majit_ir::value::Type::Float, majit_ir::value::Type::Float],
                            majit_ir::value::Type::Float,
                            EffectInfo::new(ExtraEffect::ElidableCanRaise, OopSpecIndex::None),
                        ),
                        args_i: vec![],
                        args_r: vec![],
                        args_f: vec![lhs, rhs],
                        result_kind: 'f',
                        indirect_targets: None,
                    },
                });
                ops.push(SpaceOperation {
                    result: None,
                    kind: OpKind::Live,
                });
                RewriteResult::Replace(ops)
            }
            // RPython `pair(StringRepr, StringRepr).rtype_add`
            // (`rpython/rtyper/lltypesystem/rstr.py` `ll_strconcat`)
            // lowers `s1 + s2` to a residual call to the concat helper.
            // Pyre's front-end emits a unified `BinOp { op: "add" }`
            // (Rust `+` is one AST node); over two Ref (string) operands
            // this lowers to the registered `jit_str_concat` host extern
            // (`pyre_object::unicodeobject`, address in `jit_fnaddr.rs`,
            // descriptor `OopSpecIndex::StrConcat` in
            // `STR_CONCAT_TARGETS`), assembling to the wired
            // `residual_call_r_r/iRd>r`.  Without this the op falls
            // through to the unwired `int_add/rr>r` default.
            OpKind::BinOp {
                op: binop_name,
                lhs,
                rhs,
                ..
            } if binop_name == "add"
                && self.get_value_kind_var(lhs) == 'r'
                && self.get_value_kind_var(rhs) == 'r' =>
            {
                let target = CallTarget::function_path(["jit_str_concat"]);
                let (funcptr, funcptr_op) = self.direct_funcptr_value(graph, &target);
                let mut ops = vec![funcptr_op];
                ops.push(SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::CallResidual {
                        funcptr: CallFuncPtr::Value(funcptr),
                        descriptor: CallDescriptor::from_signature(
                            &[majit_ir::value::Type::Ref, majit_ir::value::Type::Ref],
                            majit_ir::value::Type::Ref,
                            EffectInfo::new(ExtraEffect::ElidableCanRaise, OopSpecIndex::StrConcat),
                        ),
                        args_i: vec![],
                        args_r: vec![lhs.clone(), rhs.clone()],
                        args_f: vec![],
                        result_kind: 'r',
                        indirect_targets: None,
                    },
                });
                ops.push(SpaceOperation {
                    result: None,
                    kind: OpKind::Live,
                });
                RewriteResult::Replace(ops)
            }
            OpKind::UnaryOp {
                op: unop_name,
                operand,
                ..
            } if unop_name == "neg" && self.get_value_kind_var(operand) == 'f' => {
                self.stamp_value_kind(
                    graph,
                    op.result.clone(),
                    crate::codewriter::type_state::ConcreteType::Float,
                );
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::UnaryOp {
                        op: "float_neg".into(),
                        operand: operand.clone(),
                        result_ty: ValueType::Float,
                    },
                }])
            }
            // RPython hits Python-`%` / `//` semantics through TWO
            // distinct routes upstream:
            //
            //   (a) the bare-op `rewrite_op_int_floordiv =
            //       _do_builtin_call` / `rewrite_op_int_mod =
            //       _do_builtin_call` rewrite at
            //       `jtransform.py:576-577`, which replaces the
            //       SpaceOperation with `direct_call(_ll_2_int_floordiv,
            //       ...)` / `direct_call(_ll_2_int_mod, ...)` BEFORE
            //       jitcode emission.  `_do_builtin_call` resolves the
            //       helper through `support.py:266 _ll_2_int_mod` /
            //       `:255 _ll_2_int_floordiv` and binds the function
            //       pointer via `support.builtin_func_for_spec`.  The
            //       post-rewrite call carries NO oopspec markup
            //       (`_ll_2_int_*` have no `@oopspec` decorator) and
            //       the helper output is C-TRUNCATING (the no-branch
            //       reverse of `ll_int_py_div`); any Python-floor
            //       correction must come from the bytecode emitter at
            //       the BinOp callsite.
            //
            //   (b) the OS_INT_PY_MOD / OS_INT_PY_DIV oopspec at
            //       `jtransform.py:2043`, reached when the rtyper
            //       directly emits `int_py_mod` / `int_py_div` ops
            //       (typically via `objspace/std/intobject.py` Python-
            //       semantic `%` / `//`).  This route stamps
            //       `int.py_mod` / `int.py_div` oopspec on the
            //       residual call so optimisations
            //       (`rewrite.py:713-766 optimize_call_int_py_div`,
            //       `intbounds.py:1654 postprocess_int_floordiv`)
            //       recognise it, and the helper output is
            //       PYTHON-FLOOR (`rint.py:399-500 ll_int_py_div` /
            //       `ll_int_py_mod`).
            //
            // Pyre's front-end emits `BinOp { op: "mod" | "floordiv" }`
            // over `i64` operands from Rust's `%` / `/` — the C-style
            // truncating primitive of the Rust language, lowered from
            // helpers like `pyre-interpreter/src/baseobjspace.rs::int_mod`
            // whose body computes `let r = va % vb; if r != 0 &&
            // (r ^ vb) < 0 { r + vb } else { r }` to convert the
            // C-trunc step into Python-floor.  The route-(a) match —
            // C-truncating input semantic, no oopspec markup — is the
            // structural fit, so the rewrite emits a `CallResidual` to
            // `_ll_2_int_mod` / `_ll_2_int_floordiv` (registered at
            // `pyre/jit_fnaddr.rs` with the canonical RPython helper
            // names; bodies in `majit_metainterp::blackhole::_ll_2_int_*`
            // reduce to `wrapping_rem` / `wrapping_div`) with
            // `OopSpecIndex::None` and `ExtraEffect::CannotRaise`.
            //
            // Effect parity: upstream `_do_builtin_call` does NOT
            // grant `EF_ELIDABLE_*` to helpers that lack the
            // `@elidable` annotator decoration — the residual call's
            // effect family is inherited from the function's actual
            // RPython annotation, not synthesised by `_do_builtin_call`.
            // `support.py:255-271 _ll_2_int_floordiv` / `_ll_2_int_mod`
            // carry no such decorator (compare `rint.py:496
            // @jit.oopspec("int.py_mod")` which DOES decorate the
            // Python-floor sibling).  Pyre therefore stamps
            // `CannotRaise`, not `ElidableCannotRaise`: the C-trunc
            // helpers do not raise (panics on `y == 0` are unreachable
            // from the trace path, gated by the wrapper's pre-check at
            // the BinOp callsite), but the JIT must NOT assume the
            // call is pure (would license CSE / DCE the upstream effect
            // family does not).
            //
            // Pre-existing optimisation passes
            // (`optimize_call_int_py_mod` /
            // `optimize_call_int_py_div` at `optimizeopt/rewrite.rs:1788,1848`)
            // stay parked for the future route-(b) path: pyre has no
            // Python-bytecode emitter that produces `int.py_mod` /
            // `int.py_div` oopspec calls today, so those passes are
            // dormant.  Performance recovery for the BinOp{mod,Int}
            // path lands when (and only when) a route-(a) optimization
            // pass is ported on top of the C-trunc helper.
            //
            // Without this rewrite the assembler encoder
            // (`codewriter/assembler.rs:2778-2789
            // `format!("int_{op}")``) would emit the bare opname,
            // leaking `int_mod/ii>i` / `int_floordiv/ii>i` into
            // `pipeline.insns` where no blackhole handler exists.
            OpKind::BinOp {
                op: binop_name,
                lhs,
                rhs,
                result_ty,
            } if matches!(binop_name.as_str(), "mod" | "floordiv" | "div")
                && matches!(self.get_value_kind_var(lhs), 'i' | 'r')
                && matches!(self.get_value_kind_var(rhs), 'i' | 'r')
                && self.binop_result_is_int(op.result.as_ref(), result_ty) =>
            {
                // **Rust low-level → RPython low-level.**  Pyre's
                // `BinOp { op: "mod" | "floordiv" | "div" }` over i64
                // operands comes from Rust's `%` / `/` operators,
                // which are C-truncating primitives.  At the IR level
                // these match RPython's low-level
                // `llop.int_mod` / `llop.int_floordiv` — the C-trunc
                // ops that the rtyper emits and that
                // `support.py:255-271 _ll_2_int_floordiv` /
                // `_ll_2_int_mod` are the no-branch helpers for
                // (`_ll_2_*` are used "only if the RPython program
                // uses `llop.int_floordiv()` explicitly", per the
                // upstream comment).
                //
                // **NOT** Python `%` / `//` / `/` parity.  High-level
                // Python ops are handled at the rtyper layer:
                // `rint.py:246-262 rtype_mod` / `rtype_floordiv` (and
                // their `rtype_inplace_*` siblings, plus
                // `rtype_div = rtype_floordiv` for integer `/`) call
                // `_rtype_call_helper(hop, 'py_mod'/'py_div', [...])`
                // which invokes the *Python-floor* helpers
                // `ll_int_py_mod` / `ll_int_py_div`
                // (`rint.py:399-500`).  Pyre is NOT porting that
                // path — it is mapping Rust's C-trunc primitives to
                // the RPython C-trunc primitives.
                //
                // Pyre carries the same `int_div`-as-`int_floordiv`
                // collapse at this layer: the rtyper-equivalent never
                // produces an `int_div` op for integer operands
                // (there is no such llop), so pyre routes
                // `BinOp { op:"div" }` through the same
                // `_ll_2_int_floordiv` residual as the `floordiv`
                // canonical.
                //
                // The gate checks the result's proven concretetype
                // explicitly so a Ref/Float/State-typed BinOp does not
                // slip through and emit an `int_mod/ii>i` opname onto
                // a non-int SSA result; when AST lowering left
                // `result_ty == Unknown`, the rtyper-equivalent
                // `type_state` is the carrier for `op.result.concretetype`.
                let _ = result_ty;
                let helper_key = if binop_name == "mod" {
                    "mod"
                } else {
                    "floordiv"
                };
                // TODO: no direct RPython precedent:
                // pyre-side recovery when an explicit
                // `lltype.cast_ptr_to_int` (`rbuiltin.py:543-548
                // genop('cast_ptr_to_int', vlist, resulttype=Signed)`)
                // emitted by the front-end is elided from the SSA chain
                // before reaching jtransform.  The gate accepts a
                // Ref-typed LHS/RHS and rebuilds the missing cast here
                // so the residual call sees Signed operands.  RPython
                // `rint.py:246-262 rtype_mod` does NOT auto-cast
                // arbitrary Ref operands; its `hop.inputargs(repr,
                // repr)` assumes the rtyper has already inserted
                // explicit casts at lltype / rbuiltin boundaries.  The
                // wider tolerance here is therefore strictly broader
                // than the RPython contract and exists only to keep the
                // dispatch table closed while the upstream cast-
                // elision is traced and fixed (the convergence path is
                // to find which simplify / inline pass drops the cast
                // and preserve it instead, then narrow this gate back
                // to `'i' && 'i'`).
                let (lhs_var, lhs_pre_ops) = self.coerce_operand_to_int(graph, lhs);
                let (rhs_var, rhs_pre_ops) = self.coerce_operand_to_int(graph, rhs);
                let mut ops = Vec::with_capacity(lhs_pre_ops.len() + rhs_pre_ops.len() + 2);
                ops.extend(lhs_pre_ops);
                ops.extend(rhs_pre_ops);
                ops.extend(self.emit_int_mod_or_floordiv_residual(
                    graph,
                    helper_key,
                    &lhs_var,
                    &rhs_var,
                    op.result.clone(),
                ));
                RewriteResult::Replace(ops)
            }
            // RPython `Transformer.rewrite_op_float_is_true(self, op)`
            // (`jtransform.py:1627-1631`):
            //
            //     def rewrite_op_float_is_true(self, op):
            //         op1 = SpaceOperation('float_ne',
            //                              [op.args[0], Constant(0.0, lltype.Float)],
            //                              op.result)
            //         return self.rewrite_operation(op1)
            //
            // Two upstream surfaces both lower to this rewrite in pyre:
            //
            //   1. The front-end emits the un-rtyped `OpKind::UnaryOp
            //      { op: "bool", .. }` over a float-kind operand
            //      (RPython `op.bool` before the rtyper).
            //   2. The rtyper itself emits `OpKind::UnaryOp { op:
            //      "float_is_true", .. }` from `FloatRepr.rtype_bool`
            //      (`rfloat.rs:191-198`, mirror of upstream
            //      `rfloat.py:rtype_bool`).
            //
            // Both shapes must be rewritten here.  If neither is
            // caught the assembler emits a literal `float_is_true`
            // opname, but downstream backends only register
            // `float_ne` — RPython jtransform.py:1627 collapses both
            // surfaces to the same canonical shape.  Pyre's rewriter
            // does not chain back into `rewrite_operation` the way
            // upstream does (the loop at `jtransform.rs:446-462`
            // consumes `Replace(ops)` without re-dispatch), so emit
            // the canonical `float_ne` opname here rather than
            // leaving an intermediate op for the float-comparison
            // arm at `jtransform.rs:827-854`.
            OpKind::UnaryOp {
                op: unop_name,
                operand,
                ..
            } if (unop_name == "bool" && self.get_value_kind_var(operand) == 'f')
                || unop_name == "float_is_true" =>
            {
                self.stamp_value_kind(
                    graph,
                    op.result.clone(),
                    crate::codewriter::type_state::ConcreteType::Signed,
                );
                let zero_var = self.fresh_synthetic_variable_typed(
                    graph,
                    crate::codewriter::type_state::ConcreteType::Float,
                );
                let zero_op = SpaceOperation {
                    result: Some(zero_var.clone()),
                    kind: OpKind::ConstFloat(0.0_f64.to_bits()),
                };
                let ne_op = SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::BinOp {
                        op: "float_ne".into(),
                        lhs: operand.clone(),
                        rhs: zero_var,
                        result_ty: ValueType::Int,
                    },
                };
                RewriteResult::Replace(vec![zero_op, ne_op])
            }
            // RPython `jtransform.py:587-588`:
            //   rewrite_op_cast_float_to_uint  = _do_builtin_call
            //   rewrite_op_cast_uint_to_float  = _do_builtin_call
            // `_do_builtin_call` (`jtransform.py:587-588`) re-routes
            // through support helpers
            // (`rpython/jit/codewriter/support.py:274 _ll_1_cast_*`)
            // so blackhole never sees a bare `cast_*_to_uint*` opname
            // — instead a `direct_call(<helper-funcptr>)` residual
            // call carries unsigned-domain semantics (e.g.
            // `r_uint(long(f))` mod 2^64 wrap, `float(u64_value)`
            // u64-domain rounding) preserved at runtime.
            //
            // Pyre's rtyper-side `rtype_cast_uint_to_float` /
            // `rtype_cast_float_to_uint`
            // (`rbuiltin.rs::rtype_cast_uint_to_float` line 1217 and
            // `rtype_cast_float_to_uint` line 1233) emit the literal
            // opname.  The signed-domain `cast_int_to_float` /
            // `cast_float_to_int` backend instructions do NOT
            // preserve the unsigned semantics: `f as i64 as f64`
            // rounds for the signed range, not the unsigned range
            // (e.g. `u64::MAX` becomes `-1.0` instead of `~1.84e19`),
            // and `f as i64` saturates outside `[-2^63, 2^63)` rather
            // than wrapping mod 2^64.
            //
            // The arms below synthesise the `direct_call` shape
            // matching `_do_builtin_call`:
            //   - `direct_funcptr_value` produces a `ConstInt` of the
            //     helper's runtime fnaddr (registered by
            //     `pyre/pyre-interpreter/src/jit_fnaddr.rs` →
            //     `majit_metainterp::blackhole::cast_*_to_*`).
            //   - `OpKind::CallResidual` mirrors
            //     `residual_call_irf_<f|i>` (`handle_residual_call` at
            //     `jtransform.py:439-470` for the integer / float
            //     register classes).
            //   - `ElidableCannotRaise` + `OopSpecIndex::None` keeps
            //     the call out of the `may_call_jitcodes` /
            //     `calldescr_canraise` set, so no `-live-` is
            //     appended (`test_flatten.py:1007-1023`).
            // Const-fold path lives in `opimpl.rs::op_cast_*`; the
            // runtime helpers reproduce the same IEEE-754 mantissa
            // decomposition so runtime and const-fold agree.
            //
            // The rich-graph compatibility path in
            // `rewrite_op_direct_call` projects `float(r_uint)` onto this
            // low-level opname, matching the real rtyper's
            // `IntegerRepr.rtype_float` conversion.  Once production
            // jtransform consumes the already-rtyped flowspace graph, that
            // projection disappears and this arm receives the rtyper op
            // directly.
            OpKind::UnaryOp {
                op: unop_name,
                operand,
                ..
            } if unop_name == "cast_uint_to_float" => {
                // `jtransform.py:587 _do_builtin_call` routes through
                // `support.py:274 _ll_1_cast_uint_to_float`.  Helper
                // `blackhole.rs::cast_uint_to_float` carries u64-domain
                // rounding matching `opimpl.rs::op_cast_uint_to_float`.
                // `handle_residual_call` (`jtransform.py:456-470`) only
                // appends `-live-` when `may_call_jitcodes` or the
                // descriptor can raise; this helper is
                // `ElidableCannotRaise` with `OopSpecIndex::None` so the
                // flatten is `residual_call_irf_f ... -> %f0` /
                // `float_return %f0` with no intervening `-live-`
                // (`test_flatten.py:1021-1023`).
                let target = CallTarget::function_path(["cast_uint_to_float"]);
                let (funcptr, funcptr_op) = self.direct_funcptr_value(graph, &target);
                self.stamp_value_kind(
                    graph,
                    op.result.clone(),
                    crate::codewriter::type_state::ConcreteType::Float,
                );
                let ops = vec![
                    funcptr_op,
                    SpaceOperation {
                        result: op.result.clone(),
                        kind: OpKind::CallResidual {
                            funcptr: CallFuncPtr::Value(funcptr),
                            descriptor: CallDescriptor::from_signature(
                                &[majit_ir::value::Type::Int],
                                majit_ir::value::Type::Float,
                                EffectInfo::new(
                                    ExtraEffect::ElidableCannotRaise,
                                    OopSpecIndex::None,
                                ),
                            ),
                            args_i: vec![operand.clone()],
                            args_r: vec![],
                            args_f: vec![],
                            result_kind: 'f',
                            indirect_targets: None,
                        },
                    },
                ];
                RewriteResult::Replace(ops)
            }
            OpKind::UnaryOp {
                op: unop_name,
                operand,
                ..
            } if unop_name == "cast_float_to_uint" => {
                // `jtransform.py:588 _do_builtin_call` →
                // `support.py:274 _ll_1_cast_float_to_uint`.  Helper
                // mirrors `opimpl.rs::op_cast_float_to_uint` (mod 2^64
                // wrap via mantissa/exponent decomposition).
                // `ElidableCannotRaise`+`OopSpecIndex::None` → no
                // `-live-` (`test_flatten.py:1007-1009`).
                let target = CallTarget::function_path(["cast_float_to_uint"]);
                let (funcptr, funcptr_op) = self.direct_funcptr_value(graph, &target);
                self.stamp_value_kind(
                    graph,
                    op.result.clone(),
                    crate::codewriter::type_state::ConcreteType::Signed,
                );
                let ops = vec![
                    funcptr_op,
                    SpaceOperation {
                        result: op.result.clone(),
                        kind: OpKind::CallResidual {
                            funcptr: CallFuncPtr::Value(funcptr),
                            descriptor: CallDescriptor::from_signature(
                                &[majit_ir::value::Type::Float],
                                majit_ir::value::Type::Int,
                                EffectInfo::new(
                                    ExtraEffect::ElidableCannotRaise,
                                    OopSpecIndex::None,
                                ),
                            ),
                            args_i: vec![],
                            args_r: vec![],
                            args_f: vec![operand.clone()],
                            result_kind: 'i',
                            indirect_targets: None,
                        },
                    },
                ];
                RewriteResult::Replace(ops)
            }
            // RPython `jtransform.py:1606` rename pass:
            //   ('uint_is_true', 'int_is_true'),
            // The Unsigned register class is 'int' at LL, so `uint_is_true`
            // is a textual alias for `int_is_true`; the backend opcode
            // table only registers `int_is_true`.  Pyre's rtyper-side
            // `rtype_uint_is_true` (`rbuiltin.rs::rtype_uint_is_true`)
            // emits the literal `uint_is_true` opname for typing
            // `r_uint(...)` truthiness; without this rename the
            // assembler emits an unmapped opname.
            OpKind::UnaryOp {
                op: unop_name,
                operand,
                ..
            } if unop_name == "uint_is_true" => {
                self.stamp_value_kind(
                    graph,
                    op.result.clone(),
                    crate::codewriter::type_state::ConcreteType::Signed,
                );
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::UnaryOp {
                        op: "int_is_true".into(),
                        operand: operand.clone(),
                        result_ty: ValueType::Int,
                    },
                }])
            }
            // pyre front-end emits assign-form binops (`add_assign`,
            // `mod_assign`, `div_assign`, ...) from Rust's `+=` / `%=`
            // / `/=` operators.  RPython has no analogue at this layer:
            // Python bytecode expands `+=` to BINARY_ADD + STORE_FAST
            // before the rtyper runs, so flow-space never sees an
            // in-place op.  Pyre canonicalises the assign form here,
            // dropping the `_assign` suffix; when the canonical name
            // matches an integer op pyre routes through a residual
            // helper above, emit the same residual directly so the
            // result does not leak past the rewrite as a bare `mod` /
            // `floordiv` / `div` op (this pass does not re-traverse
            // its own Replace output).
            //
            // **Rust low-level → RPython low-level.**  Pyre's
            // `mod_assign` / `div_assign` come from Rust's `%=` / `/=`
            // on i64 (C-trunc primitive), matching RPython's
            // `llop.int_mod` / `llop.int_floordiv` low-level llops.
            // This is **not** a port of upstream's `rtype_inplace_mod
            // = rtype_mod` / `rtype_inplace_div = rtype_inplace_floordiv`
            // — those route the *Python-level* `%=` / `/=` through
            // `_rtype_call_helper(hop, 'py_mod'/'py_div', ...)` to the
            // Python-floor `ll_int_py_mod` / `ll_int_py_div` helpers
            // (`rint.py:399-500`).  Pyre carries no Python-level
            // `%=` / `/=` at this layer.
            OpKind::BinOp {
                op: binop_name,
                lhs,
                rhs,
                result_ty,
            } if canonical_assign_binop(binop_name).is_some() => {
                let canonical =
                    canonical_assign_binop(binop_name).expect("guard checked assign binop");
                // `rint.py:253-255`: `rtype_div = rtype_floordiv` (and
                // `rtype_inplace_div = rtype_inplace_floordiv`) — integer
                // `div` collapses to `floordiv` at the rtyper layer.
                // `div_assign → "div"` from `canonical_assign_binop`,
                // then this branch treats `"div"` as `floordiv` for the
                // residual route, mirroring the plain `BinOp { op:"div" }`
                // arm above.
                let residual_key: Option<&'static str> = match canonical {
                    "mod" => Some("mod"),
                    "floordiv" | "div" => Some("floordiv"),
                    _ => None,
                };
                if let Some(key) = residual_key
                    && self.get_value_kind_var(lhs) == 'i'
                    && self.get_value_kind_var(rhs) == 'i'
                    && self.binop_result_is_int(op.result.as_ref(), result_ty)
                {
                    return RewriteResult::Replace(self.emit_int_mod_or_floordiv_residual(
                        graph,
                        key,
                        lhs,
                        rhs,
                        op.result.clone(),
                    ));
                }
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::BinOp {
                        op: canonical.to_string(),
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                        result_ty: result_ty.clone(),
                    },
                }])
            }
            _ => RewriteResult::Keep,
        }
    }

    /// RPython: `Transformer.make_three_lists(vars)` (jtransform.py:437-445).
    /// Split args into three lists by kind (int, ref, float) keyed on
    /// the backing [`crate::flowspace::model::Variable`] (orthodox per
    /// `flowspace/model.py:Variable`).
    ///
    /// RPython: `add_in_correct_list(v, lst_i, lst_r, lst_f)` checks
    /// `getkind(v.concretetype)` and appends to the matching list.
    /// Void args are skipped.
    fn make_three_lists_from_vars(
        &self,
        args: &[crate::flowspace::model::Variable],
    ) -> (
        Vec<crate::flowspace::model::Variable>,
        Vec<crate::flowspace::model::Variable>,
        Vec<crate::flowspace::model::Variable>,
    ) {
        let mut args_i = Vec::new();
        let mut args_r = Vec::new();
        let mut args_f = Vec::new();
        for var in args {
            let kind = self.get_value_kind_var(var);
            match kind {
                'i' => args_i.push(var.clone()),
                'r' => args_r.push(var.clone()),
                'f' => args_f.push(var.clone()),
                'v' => {}                      // void — skip (RPython jtransform.py:449)
                _ => args_r.push(var.clone()), // unknown → ref
            }
        }
        (args_i, args_r, args_f)
    }

    /// RPython: `getkind(v.concretetype)` — get the kind of a value.
    ///
    /// Mirrors `rpython/jit/metainterp/history.py:45-71 getkind`: GC
    /// pointers (`TYPE.TO._gckind != 'raw'`, history.py:66-67) collapse
    /// to `"ref"`, raw pointers (`gckind == 'raw'`, history.py:64-65)
    /// collapse to `"int"`, primitives map to `"int"` / `"float"` /
    /// `"void"`.  The function therefore returns ONLY one of `'i'` /
    /// `'r'` / `'f'` / `'v'`; sub-pointer distinctions like
    /// `Ptr(rstr.STR)` vs `Ptr(rstr.UNICODE)` are NOT folded into the
    /// kind char.
    ///
    /// Pyre currently has no `Ptr(rstr.STR)` / `Ptr(rstr.UNICODE)`
    /// concrete-type channel because pyre-object lacks those GC layouts.
    /// Re-introducing that distinction must happen at the hint dispatch
    /// site, mirroring upstream's explicit
    /// `op.args[0].concretetype == lltype.Ptr(rstr.STR)` checks, not by
    /// overloading `getkind`.
    ///
    /// Refining this getter to return `'s'` / `'u'` would propagate
    /// the refinement into unrelated rewrite arms
    /// (`ptr_eq`/`ptr_ne` synthesis at line 775 above,
    /// `kind_char_to_name` opname formation in `assembler.rs`,
    /// `jit.assert_green` / `jit.isconstant` / `jit.isvirtual` etc.)
    /// where RPython expects a plain `'r'`, breaking parity at every
    /// such call site.
    ///
    /// Reads the Variable's `.concretetype` cell directly per RPython
    /// `rtyper.py:258 v.concretetype = ...` parity, falling back to
    /// `'r'` when the cell is `Unknown`.
    fn get_value_kind_var(&self, var: &crate::flowspace::model::Variable) -> char {
        match FunctionGraph::concretetype_of(var) {
            crate::codewriter::type_state::ConcreteType::Signed => 'i',
            crate::codewriter::type_state::ConcreteType::GcRef => 'r',
            crate::codewriter::type_state::ConcreteType::Float => 'f',
            crate::codewriter::type_state::ConcreteType::Void => 'v',
            crate::codewriter::type_state::ConcreteType::Unknown => 'r',
        }
    }

    fn get_value_type(&self, var: &crate::flowspace::model::Variable) -> Option<ValueType> {
        // RPython `jit/codewriter/jtransform.py`: `getkind(v.concretetype)`
        // — read kind off the Variable.concretetype slot directly.
        match FunctionGraph::concretetype_of(var) {
            crate::codewriter::type_state::ConcreteType::Signed => Some(ValueType::Int),
            crate::codewriter::type_state::ConcreteType::GcRef => Some(ValueType::Ref(None)),
            crate::codewriter::type_state::ConcreteType::Float => Some(ValueType::Float),
            crate::codewriter::type_state::ConcreteType::Void => Some(ValueType::Void),
            crate::codewriter::type_state::ConcreteType::Unknown => None,
        }
    }

    fn binop_result_is_int(
        &self,
        result: Option<&crate::flowspace::model::Variable>,
        result_ty: &ValueType,
    ) -> bool {
        if *result_ty == ValueType::Int {
            return true;
        }
        let Some(var) = result else {
            return false;
        };
        self.get_value_type(var) == Some(ValueType::Int)
    }

    /// Emit the `_ll_2_int_mod` / `_ll_2_int_floordiv` residual call
    /// pair (funcptr-materialisation op + `CallResidual`) shared
    /// between the plain `mod` / `floordiv` / `div` arm and the
    /// `canonical_assign_binop` arm.  `helper_key` is `"mod"` for
    /// the int-mod residual or anything else (`"floordiv"`, `"div"`)
    /// for the int-floordiv residual — `rint.py:253-255 rtype_div =
    /// rtype_floordiv` aliases integer `div` to `floordiv` at the
    /// rtyper layer, and pyre carries that alias through here.
    ///
    /// `jtransform.py:469-470`'s `-live-` gate fires only on
    /// `may_call_jitcodes or calldescr_canraise`; this residual is
    /// neither (the helper is an `extern "C"` C-truncating arithmetic
    /// primitive flagged `LLOp(canfold=True)` upstream —
    /// `lloperation.py:203-204`), so no `OpKind::Live` follows.
    fn emit_int_mod_or_floordiv_residual(
        &mut self,
        graph: &mut FunctionGraph,
        helper_key: &str,
        lhs: &crate::flowspace::model::Variable,
        rhs: &crate::flowspace::model::Variable,
        result: Option<crate::flowspace::model::Variable>,
    ) -> Vec<SpaceOperation> {
        let helper_name = if helper_key == "mod" {
            "_ll_2_int_mod"
        } else {
            "_ll_2_int_floordiv"
        };
        let target = CallTarget::function_path([helper_name]);
        let (funcptr, funcptr_op) = self.direct_funcptr_value(graph, &target);
        let lhs_var = lhs.clone();
        let rhs_var = rhs.clone();
        let ops = vec![
            funcptr_op,
            SpaceOperation {
                result,
                kind: OpKind::CallResidual {
                    funcptr: CallFuncPtr::Value(funcptr),
                    descriptor: CallDescriptor::from_signature(
                        &[majit_ir::value::Type::Int, majit_ir::value::Type::Int],
                        majit_ir::value::Type::Int,
                        EffectInfo::new(ExtraEffect::CannotRaise, OopSpecIndex::None),
                    ),
                    args_i: vec![lhs_var, rhs_var],
                    args_r: vec![],
                    args_f: vec![],
                    result_kind: 'i',
                    indirect_targets: None,
                },
            },
        ];
        ops
    }

    /// RPython's float rtyper calls `hop.inputargs(Float, Float)`, which
    /// inserts `cast_int_to_float` for mixed int/float operands before
    /// emitting `float_*` or the `math.fmod` helper call.  Pyre's float
    /// rewrite arms only enter when both operands are
    /// `is_float_rewrite_domain` (i.e. `'i'` or `'f'`); Ref operands do
    /// not reach this helper.
    fn coerce_operand_to_float_domain(
        &mut self,
        graph: &mut FunctionGraph,
        value: &crate::flowspace::model::Variable,
    ) -> (crate::flowspace::model::Variable, Vec<SpaceOperation>) {
        match self.get_value_kind_var(value) {
            'f' => (value.clone(), Vec::new()),
            'i' => self.coerce_operand_to_float(graph, value),
            _ => (value.clone(), Vec::new()),
        }
    }

    /// RPython's float rtyper calls `hop.inputargs(Float, Float)`, which
    /// inserts `cast_int_to_float` for mixed int/float operands before
    /// emitting `float_*` or the `math.fmod` helper call.
    fn coerce_operand_to_float(
        &mut self,
        graph: &mut FunctionGraph,
        value: &crate::flowspace::model::Variable,
    ) -> (crate::flowspace::model::Variable, Vec<SpaceOperation>) {
        if self.get_value_kind_var(value) != 'i' {
            return (value.clone(), Vec::new());
        }
        let coerced = self.fresh_synthetic_variable_typed(
            graph,
            crate::codewriter::type_state::ConcreteType::Float,
        );
        (
            coerced.clone(),
            vec![SpaceOperation {
                result: Some(coerced),
                kind: OpKind::UnaryOp {
                    op: "cast_int_to_float".into(),
                    operand: value.clone(),
                    result_ty: ValueType::Float,
                },
            }],
        )
    }

    /// TODO: recovery helper — no direct RPython
    /// precedent.  Inserts an explicit `cast_ptr_to_int` op for a
    /// Ref-typed operand reaching an arithmetic site that requires
    /// Int operands.  Upstream RPython does NOT auto-cast arbitrary
    /// Ref to Signed at the rtyper boundary; ptr→int conversions come
    /// from explicit `lltype.cast_ptr_to_int` builtins emitted at the
    /// rbuiltin layer.  Pyre's analyzer/simplify chain may elide an
    /// emitted cast before jtransform sees it, leaving a bare Ref at
    /// the binop callsite; this helper rebuilds the missing cast so
    /// the residual call sees Signed operands.  The convergence path
    /// is to fix the cast elision upstream and retire this helper.
    /// Non-Ref operands are returned unchanged.
    fn coerce_operand_to_int(
        &mut self,
        graph: &mut FunctionGraph,
        value: &crate::flowspace::model::Variable,
    ) -> (crate::flowspace::model::Variable, Vec<SpaceOperation>) {
        if self.get_value_kind_var(value) != 'r' {
            return (value.clone(), Vec::new());
        }
        let coerced = self.fresh_synthetic_variable_typed(
            graph,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        (
            coerced.clone(),
            vec![SpaceOperation {
                result: Some(coerced),
                kind: OpKind::UnaryOp {
                    op: "cast_ptr_to_int".into(),
                    operand: value.clone(),
                    result_ty: ValueType::Int,
                },
            }],
        )
    }

    /// RPython: `Transformer.rewrite_op_hint(op)`.
    /// Dispatches based on the hint kind (access_directly, force_virtualizable,
    /// fresh_virtualizable, promote, etc.)
    fn rewrite_op_hint(
        &mut self,
        op: &SpaceOperation,
        hint_kind: crate::hints::HintKind,
        args: &[crate::flowspace::model::Variable],
        label: &str,
        graph_name: &str,
    ) -> RewriteResult {
        match hint_kind {
            crate::hints::HintKind::AccessDirectly | crate::hints::HintKind::FreshVirtualizable => {
                // `jtransform.py:655-656 if hints.get('fresh_virtualizable')
                // or hints.get('access_directly'): return` — both are
                // consumed without emitting an operation.
                //
                // `fresh_virtualizable` additionally has to be remembered:
                // upstream carries it as an annotator flag on the instance
                // (`rlib/jit.py:321-322`), which `rvirtualizable.py:49-53
                // hook_access_field` re-materialises as the third argument
                // of a `jit_force_virtualizable` in front of every
                // redirected-field access, and `jtransform.py:2171
                // rewrite_op_jit_force_virtualizable` files under
                // `vable_flags[v_inst]`.  Pyre has neither the instance
                // flag nor `hook_access_field`, so the hint call is the
                // only carrier and files the flag directly against the
                // virtualizable it names.
                //
                // That substitution is weaker than what it replaces, and
                // the gap is per-block.  `vable_flags` is rebuilt at the
                // top of every block (`jtransform.py:76-77`); upstream
                // tolerates that because it re-materialises the entry in
                // front of *each* redirected access, so its flag holds
                // wherever the access happens to sit.  A single hint call
                // cannot: it covers only the accesses in its own block.
                // A redirected field access on the same instance in any
                // other block — the far side of a branch, a loop body, an
                // assertion guarded by a null test — still lowers to
                // `getfield_vable_*` / `setfield_vable_*` and still admits
                // an array field to `vable_array_vars`.  The fix for a
                // surviving access is another hint call in the block that
                // holds it, not a change here.
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!("rewrite: {label}(...) → identity"),
                });
                if let Some(arg) = args.first() {
                    if hint_kind == crate::hints::HintKind::FreshVirtualizable {
                        // Key on the resolved operand: `optimize_block`
                        // remaps every op through `aliases` before rewriting,
                        // so this is the same handle the later field reads
                        // present as their base.
                        self.vable_flags.insert(
                            resolve_alias(arg, &self.aliases),
                            VableFlag::FreshVirtualizable,
                        );
                    }
                    RewriteResult::Identity(arg.clone())
                } else {
                    RewriteResult::Keep
                }
            }
            crate::hints::HintKind::ForceVirtualizable => {
                // RPython: emit hint_force_virtualizable, preserve value as identity
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!("rewrite: {label}(...) → VableForce"),
                });
                self.vable_rewrites += 1;
                if let Some(arg) = args.first() {
                    let base = resolve_alias(arg, &self.aliases);
                    if let Some(result) = op.result.clone() {
                        self.aliases.insert(result, base.clone());
                    }
                    RewriteResult::Replace(vec![SpaceOperation {
                        result: None,
                        kind: OpKind::VableForce { base },
                    }])
                } else {
                    RewriteResult::Keep
                }
            }
            crate::hints::HintKind::Promote => {
                // `rpython/jit/codewriter/jtransform.py:608-614`:
                //     if hints.get('promote') and op.args[0].concretetype is not lltype.Void:
                //         assert op.args[0].concretetype != lltype.Ptr(rstr.STR)
                //         kind = getkind(op.args[0].concretetype)
                //         op0 = SpaceOperation('-live-', [], None)
                //         op1 = SpaceOperation('%s_guard_value' % kind, [op.args[0]], None)
                //         # the special return value None forces op.result
                //         # to be considered equal to op.args[0]
                //         return [op0, op1, None]
                //
                // Skip void args (`concretetype is not lltype.Void` guard).
                // The string-pointer special case (`promote_string` /
                // `str_guard_value`) is a separate hint kind; this arm only
                // handles plain `int/ref/float_guard_value` shapes.  The
                // `None` sentinel that aliases the result back to the input
                // is realized in pyre by `self.aliases.insert(result, base)`
                // before emitting the replacement ops.
                self.rewrite_op_hint_guard_value_family(op, args, label, graph_name)
            }
            crate::hints::HintKind::PromoteString => {
                // `rpython/jit/codewriter/jtransform.py:615-631 promote_string`:
                //     S = lltype.Ptr(rstr.STR)
                //     assert op.args[0].concretetype == S
                //     ...register OS_STREQ_NONNULL + emit str_guard_value...
                //
                // The upstream `str_guard_value` op is the value-equality
                // promotion of an `rstr.STR` low-level string: it guards on
                // the *characters* (`support.py:526-538 _ll_2_str_eq_nonnull`)
                // so two distinct-but-equal `Ptr(rstr.STR)` operands fold to
                // one trace.  That representation is rtyper-internal; pyre
                // interpreter strings are `W_UnicodeObject` GC refs, never
                // `Ptr(rstr.STR)`, so the `assert ... == Ptr(rstr.STR)` is
                // satisfied by absence and there is no inline char array to
                // value-compare.  The faithful pyre lowering of a string
                // `promote` is therefore the ref-kind member of the
                // `<kind>_guard_value` family — `r_guard_value`, an identity
                // guard on the `W_UnicodeObject` pointer (exact for interned
                // names, sound for the rest).  `get_value_kind_var` reports
                // `'r'` for the ref operand, so the shared helper emits
                // `r_guard_value` directly.
                self.rewrite_op_hint_guard_value_family(op, args, label, graph_name)
            }
            crate::hints::HintKind::PromoteUnicode => {
                // `rpython/jit/codewriter/jtransform.py:632-648 promote_unicode`:
                //     U = lltype.Ptr(rstr.UNICODE)
                //     assert op.args[0].concretetype == U
                //     ...register OS_UNIEQ_NONNULL + emit str_guard_value...
                //
                // Same shape as `PromoteString`: the `rstr.UNICODE`
                // value-equality `str_guard_value` has no `W_UnicodeObject`
                // counterpart, so the ref operand lowers through the shared
                // `<kind>_guard_value` family to `r_guard_value`.
                self.rewrite_op_hint_guard_value_family(op, args, label, graph_name)
            }
            crate::hints::HintKind::PromoteOrString => {
                // `rpython/jit/codewriter/jtransform.py:599-606` —
                // when a `hint(arg, ...)` carries both `promote=True`
                // and `promote_string=True`, jtransform discards one
                // based on `op.args[0].concretetype`:
                //
                //     if hints.get('promote_string') and hints.get('promote'):
                //         hints = hints.copy()
                //         if op.args[0].concretetype == lltype.Ptr(rstr.STR):
                //             del hints['promote']
                //         else:
                //             del hints['promote_string']
                //
                // Pyre has no `Ptr(rstr.STR)` layout (see
                // `PromoteString` arm above), so the upstream `if`
                // branch is structurally unreachable — every dual-flag
                // hint takes the `else` branch
                // (`del hints['promote_string']`) and falls through
                // to the plain `promote` arm, emitting
                // `<kind>_guard_value` per `jit.py:608-614` +
                // `getkind(Ptr) == "ref"` (`rpython/jit/metainterp/
                // history.py:64`).
                self.rewrite_op_hint_guard_value_family(op, args, label, graph_name)
            }
        }
    }

    /// Shared body for the `promote=True` arm of
    /// `rpython/jit/codewriter/jtransform.py:608-614 rewrite_op_hint`.
    ///
    /// Returns `RewriteResult::Replace([-live-, <kind>_guard_value(arg)])`
    /// after seeding `self.aliases.insert(result, arg)` to realize the
    /// upstream `None` sentinel (`# the special return value None forces
    /// op.result to be considered equal to op.args[0]`).  The
    /// `<kind>` char is the upstream `getkind()` of `op.args[0]`
    /// (`'i'`/`'r'`/`'f'`); void args fall through.
    ///
    /// `promote_string` / `promote_unicode` (jit.py:615-648) also route
    /// through this helper: upstream's 3-input `str_guard_value` is a
    /// value-equality guard over an `rstr.STR`/`UNICODE` char array, a
    /// representation pyre interpreter strings (`W_UnicodeObject` refs) do
    /// not use, so the string promote collapses to the ref-kind member of
    /// this family — `r_guard_value`.
    fn rewrite_op_hint_guard_value_family(
        &mut self,
        op: &SpaceOperation,
        args: &[crate::flowspace::model::Variable],
        label: &str,
        graph_name: &str,
    ) -> RewriteResult {
        let Some(arg) = args.first() else {
            return RewriteResult::Keep;
        };
        let base = resolve_alias(arg, &self.aliases);
        let kind_char = self.get_value_kind_var(&base);
        // jtransform.py:608 `op.args[0].concretetype is not lltype.Void`
        // guard — void args fall through the rewrite (caller may want
        // to keep the original op).
        if kind_char == 'v' {
            return RewriteResult::Keep;
        }
        // jtransform.py:609 `assert op.args[0].concretetype !=
        // lltype.Ptr(rstr.STR)` — pyre has no `Ptr(rstr.STR)` GC
        // layout (`rpython/rtyper/lltypesystem/rstr.py:1226-1237`), so the
        // upstream assertion is structurally satisfied by absence: no
        // pyre value can carry that concretetype, hence no `Ptr(STR)`
        // operand can reach this arm.  Re-introduce the assertion
        // once pyre-object grows the layout and the `PromoteString` /
        // `PromoteUnicode` arms can satisfy their upstream asserts.
        if let Some(result) = op.result.clone() {
            self.aliases.insert(result, base.clone());
        }
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("rewrite: {label}(...) → {kind_char}_guard_value"),
        });
        RewriteResult::Replace(vec![
            SpaceOperation {
                result: None,
                kind: OpKind::Live,
            },
            SpaceOperation {
                result: None,
                kind: OpKind::GuardValue {
                    value: base,
                    kind_char,
                },
            },
        ])
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
        let inline_substruct_offset = self
            .callcontrol
            .as_deref()
            .and_then(|cc| crate::assembler::inline_substruct_field_offset(cc, field));
        if let Some(0) = inline_substruct_offset {
            let OpKind::FieldRead { base, .. } = &op.kind else {
                unreachable!("rewrite_op_getfield called on non-FieldRead op")
            };
            return RewriteResult::Identity(base.clone());
        }
        if inline_substruct_offset.is_some()
            && op
                .result
                .as_ref()
                .is_some_and(|result| self.call_argument_vars.contains(result))
        {
            // `jtransform.py:945-946 rewrite_op_getsubstruct` refuses GC
            // substructures during translation. Pyre must still resolve the
            // portal graph, so defer the same refusal to a result-less runtime
            // abort; the enclosing call then remains residual.
            return RewriteResult::Replace(vec![
                SpaceOperation {
                    result: None,
                    kind: OpKind::Abort {
                        kind: crate::model::UnknownKind::UnsupportedExpr {
                            variant: crate::model::UnsupportedExprKind::RawAddr,
                        },
                    },
                },
                op.clone(),
            ]);
        }
        let typed_ty = op
            .result
            .as_ref()
            .and_then(|result| self.get_value_type(result))
            .unwrap_or_else(|| ty.clone());
        // `jtransform.py:990-993` — a just-allocated virtualizable is not
        // under the virtualizable protocol yet, so neither the array
        // tracking nor the scalar lowering below may fire for it.
        let fresh_virtualizable = match &op.kind {
            OpKind::FieldRead { base, .. } => self.is_fresh_virtualizable(base),
            _ => unreachable!("rewrite_op_getfield called on non-FieldRead op"),
        };
        // Two more suppressions, both reached structurally.  A container
        // that was not dereferenced is a value in this frame, so the
        // access cannot be to a live virtualizable — that is what covers
        // the frame constructor, where the hint upstream relies on cannot
        // land.  And a projection whose *address* was taken is not a read
        // of the field at all; tracking it would drop the op and leave the
        // address's consumer with an undefined operand.
        let fresh_virtualizable = fresh_virtualizable || field.suppresses_virtualizable();
        // Track virtualizable array field reads
        if let Some(array_field) = self
            .config
            .vable_arrays
            .iter()
            .find(|c| c.matches(field))
            .filter(|_| !fresh_virtualizable)
            && let Some(result) = op.result.clone()
        {
            // RPython: vable_array_vars[result] = (v_base, arrayfielddescr, arraydescr)
            // We store the vable base plus the arraydescr properties.
            let base_var = match &op.kind {
                OpKind::FieldRead { base, .. } => base.clone(),
                _ => unreachable!("rewrite_op_getfield called on non-FieldRead op"),
            };
            // Vable arrays hold word-wide `PyObjectRef` items; the
            // default must be the target word, not the build host's.
            let itemsize = array_field
                .array_itemsize
                .unwrap_or_else(crate::layout::target_word_size);
            let is_signed = array_field.array_is_signed.unwrap_or(false);
            self.vable_array_vars
                .insert(result, (base_var, array_field.index, itemsize, is_signed));
        }
        // Virtualizable scalar field → VableFieldRead
        if let Some(vable_field) = self
            .config
            .vable_fields
            .iter()
            .find(|c| c.matches(field))
            .filter(|_| !fresh_virtualizable)
        {
            self.notes.push(GraphTransformNote {
                function: graph_name.to_string(),
                detail: format!(
                    "rewrite: {} → VableFieldRead[{}]",
                    field.name, vable_field.index
                ),
            });
            self.vable_rewrites += 1;
            let base_var = match &op.kind {
                OpKind::FieldRead { base, .. } => base.clone(),
                _ => unreachable!("rewrite_op_getfield called on non-FieldRead op"),
            };
            // `jtransform.py:845-847` returns a two-element list led by
            // `-live-`: a virtualizable access is a valid resume point, so
            // the liveness marker belongs in front of it.
            return RewriteResult::Replace(vec![
                SpaceOperation {
                    result: None,
                    kind: OpKind::Live,
                },
                SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::VableFieldRead {
                        base: base_var,
                        field_index: vable_field.index,
                        ty: typed_ty.clone(),
                    },
                },
            ]);
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
            let base = base.clone();
            if rank.is_quasi_immutable() {
                // TODO: RPython
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
                            base: base.clone(),
                            field: field.clone(),
                            mutate_field,
                        },
                    },
                    SpaceOperation {
                        result: op.result.clone(),
                        kind: OpKind::FieldRead {
                            base: base.clone(),
                            field: field.clone(),
                            ty: typed_ty.clone(),
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
                    result: op.result.clone(),
                    kind: OpKind::FieldRead {
                        base: base.clone(),
                        field: field.clone(),
                        ty: typed_ty.clone(),
                        pure: true,
                    },
                }]);
            }
        }
        if &typed_ty != ty {
            let OpKind::FieldRead { base, pure, .. } = &op.kind else {
                return RewriteResult::Keep;
            };
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::FieldRead {
                    base: base.clone(),
                    field: field.clone(),
                    ty: typed_ty,
                    pure: *pure,
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
        value: &LinkArg,
        ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        // `jtransform.py:912 self._check_no_vable_array(op.args)` —
        // upstream's `op.args` is `[v_inst, c_fieldname, v_value]`; the
        // field name rides in a descriptor here, leaving the base and the
        // stored value as the operands.
        //
        if let OpKind::FieldWrite { base, .. } = &op.kind {
            self.check_no_vable_array(
                std::iter::once(base).chain(value.as_variable()),
                graph_name,
                "setfield operand",
            );
        }
        let typed_ty = value
            .as_variable()
            .and_then(|v| self.get_value_type(v))
            .unwrap_or_else(|| ty.clone());
        // `jtransform.py:923 if self.is_virtualizable_getset(op)` — the
        // write side consults the same predicate, so the
        // `jtransform.py:990-993` fresh-virtualizable arm suppresses it too.
        let fresh_virtualizable = match &op.kind {
            OpKind::FieldWrite { base, .. } => self.is_fresh_virtualizable(base),
            _ => unreachable!("rewrite_op_setfield called on non-FieldWrite op"),
        };
        // The structural arm, as on the read side: a write into a
        // non-dereferenced local aggregate is not a virtualizable write.
        let fresh_virtualizable = fresh_virtualizable || field.base_is_local_aggregate();
        if let Some(vable_field) = self
            .config
            .vable_fields
            .iter()
            .find(|c| c.matches(field))
            .filter(|_| !fresh_virtualizable)
        {
            self.notes.push(GraphTransformNote {
                function: graph_name.to_string(),
                detail: format!(
                    "rewrite: {} = ... → VableFieldWrite[{}]",
                    field.name, vable_field.index
                ),
            });
            self.vable_rewrites += 1;
            let base_var = match &op.kind {
                OpKind::FieldWrite { base, .. } => base.clone(),
                _ => unreachable!("rewrite_op_setfield called on non-FieldWrite op"),
            };
            // `jtransform.py:926-928` — `-live-` leads the virtualizable
            // write for the same reason as the read.
            return RewriteResult::Replace(vec![
                SpaceOperation {
                    result: None,
                    kind: OpKind::Live,
                },
                SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::VableFieldWrite {
                        base: base_var,
                        field_index: vable_field.index,
                        // `rewrite_op_setfield` forwards `v_value` unchanged to
                        // `setfield_vable_%s` (`jtransform.py:921-927`); a
                        // constant operand stays inline.  `setfield_vable_i`
                        // is not in USE_C_FORM, so the assembler encodes it as
                        // a pool `i` slot rather than the short `c` byte.
                        value: value.clone(),
                        ty: typed_ty,
                    },
                },
            ]);
        }
        if &typed_ty != ty {
            let base_var = match &op.kind {
                OpKind::FieldWrite { base, .. } => base.clone(),
                _ => unreachable!("rewrite_op_setfield called on non-FieldWrite op"),
            };
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::FieldWrite {
                    base: base_var,
                    field: field.clone(),
                    value: value.clone(),
                    ty: typed_ty,
                },
            }]);
        }
        RewriteResult::Keep
    }

    /// RPython: rewrite_op_getarrayitem
    fn rewrite_op_getarrayitem(
        &mut self,
        op: &SpaceOperation,
        base: &crate::flowspace::model::Variable,
        index: &crate::flowspace::model::Variable,
        item_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        let typed_item_ty = op
            .result
            .as_ref()
            .and_then(|result| self.get_value_type(result))
            .unwrap_or_else(|| item_ty.clone());
        if let Some((vable_base, arr_idx, itemsize, is_signed)) =
            self.vable_array_vars.get(base).cloned()
        {
            self.notes.push(GraphTransformNote {
                function: graph_name.to_string(),
                detail: format!("rewrite: array[idx] → VableArrayRead[{arr_idx}]"),
            });
            self.vable_rewrites += 1;
            // `jtransform.py:764-767` — `-live-` leads the virtualizable
            // array read.
            return RewriteResult::Replace(vec![
                SpaceOperation {
                    result: None,
                    kind: OpKind::Live,
                },
                SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::VableArrayRead {
                        base: vable_base,
                        array_index: arr_idx,
                        elem_index: index.clone(),
                        item_ty: typed_item_ty,
                        array_itemsize: itemsize,
                        array_is_signed: is_signed,
                    },
                },
            ]);
        }
        if &typed_item_ty != item_ty {
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::ArrayRead {
                    base: base.clone(),
                    index: index.clone(),
                    item_ty: typed_item_ty,
                    array_type_id: match &op.kind {
                        OpKind::ArrayRead { array_type_id, .. } => array_type_id.clone(),
                        _ => unreachable!("rewrite_op_getarrayitem called on non-ArrayRead op"),
                    },
                    nolength: match &op.kind {
                        OpKind::ArrayRead { nolength, .. } => *nolength,
                        _ => unreachable!("rewrite_op_getarrayitem called on non-ArrayRead op"),
                    },
                    // Preserve the source foldable/immutable flag through
                    // the item_ty re-type — never hardcode it (rlist.py:724
                    // ll_getitem_foldable_nonneg).
                    pure: match &op.kind {
                        OpKind::ArrayRead { pure, .. } => *pure,
                        _ => unreachable!("rewrite_op_getarrayitem called on non-ArrayRead op"),
                    },
                },
            }]);
        }
        RewriteResult::Keep
    }

    /// RPython: rewrite_op_setarrayitem
    fn rewrite_op_setarrayitem(
        &mut self,
        op: &SpaceOperation,
        base: &crate::flowspace::model::Variable,
        index: &crate::flowspace::model::Variable,
        value: &crate::model::LinkArg,
        item_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        let typed_item_ty = value
            .as_variable()
            .and_then(|v| self.get_value_type(v))
            .unwrap_or_else(|| item_ty.clone());
        if let Some((vable_base, arr_idx, itemsize, is_signed)) =
            self.vable_array_vars.get(base).cloned()
        {
            self.notes.push(GraphTransformNote {
                function: graph_name.to_string(),
                detail: format!("rewrite: array[idx] = v → VableArrayWrite[{arr_idx}]"),
            });
            self.vable_rewrites += 1;
            // `jtransform.py:798-801` — `-live-` leads the virtualizable
            // array write.
            return RewriteResult::Replace(vec![
                SpaceOperation {
                    result: None,
                    kind: OpKind::Live,
                },
                SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::VableArrayWrite {
                        base: vable_base,
                        array_index: arr_idx,
                        elem_index: index.clone(),
                        // The vable rewrite only fires for virtualizable
                        // arrays, which never carry an inline-const store; a
                        // VableArrayWrite value is always a register.
                        value: value
                            .as_variable()
                            .expect("vable array writes carry a Variable value")
                            .clone(),
                        item_ty: typed_item_ty,
                        array_itemsize: itemsize,
                        array_is_signed: is_signed,
                    },
                },
            ]);
        }
        if &typed_item_ty != item_ty {
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::ArrayWrite {
                    base: base.clone(),
                    index: index.clone(),
                    value: value.clone(),
                    item_ty: typed_item_ty,
                    array_type_id: match &op.kind {
                        OpKind::ArrayWrite { array_type_id, .. } => array_type_id.clone(),
                        _ => unreachable!("rewrite_op_setarrayitem called on non-ArrayWrite op"),
                    },
                    nolength: match &op.kind {
                        OpKind::ArrayWrite { nolength, .. } => *nolength,
                        _ => unreachable!("rewrite_op_setarrayitem called on non-ArrayWrite op"),
                    },
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
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
        graph: &mut crate::model::FunctionGraph,
    ) -> RewriteResult {
        // RPython `jtransform.py:406-408`:
        //   def rewrite_op_direct_call(op): ... handle_%s_call
        //
        // The indirect path (`jtransform.py:410-412 rewrite_op_indirect_call`)
        // is reached via `OpKind::IndirectCall` which the rtyper-equivalent
        // layer (`translator/rtyper/rpbc.rs::lower_indirect_calls`) emits
        // before jtransform runs, dispatched from `rewrite_operation` to
        // `lower_indirect_call_op`. By this point, no `CallTarget::Indirect`
        // can survive — that invariant is asserted in debug builds by
        // `assert_no_indirect_call_targets`.
        debug_assert!(
            !matches!(target, CallTarget::Indirect { .. }),
            "CallTarget::Indirect must be lowered by translator/rtyper/rpbc.rs \
             before reaching rewrite_op_direct_call",
        );
        // RPython `IntegerRepr.rtype_float` (`rint.py:144-147`) converts an
        // Unsigned input to Float, emitting `cast_uint_to_float`; jtransform
        // then applies `_do_builtin_call` (`jtransform.py:587`) and reaches
        // `support.py:274 _ll_1_cast_uint_to_float`.
        //
        // Production still feeds jtransform the rich MIR graph while the
        // real rtyper specializes an ephemeral flowspace graph (see
        // `jtransform_shadow.rs`).  Project this one source-level
        // `float(r_uint)` call to the exact low-level op the rtyper produced,
        // then dispatch through the ordinary rewrite arm above.  This keeps
        // the helper address/effect descriptor and unsigned rounding on the
        // upstream path; it is removed when jtransform consumes the rtyped
        // graph directly.
        if matches!(target, CallTarget::FunctionPath { segments } if segments == &["float"])
            && args.len() == 1
            && matches!(result_ty, ValueType::Float)
            && variable_has_declared_unsigned_type(graph, &args[0])
        {
            let cast_op = SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::UnaryOp {
                    op: "cast_uint_to_float".into(),
                    operand: args[0].clone(),
                    result_ty: ValueType::Float,
                },
            };
            return self.rewrite_operation(&cast_op, graph_name, graph);
        }
        // RPython `jtransform.py:1658-1663 rewrite_op_jit_marker`:
        // marker calls never reach `guess_call_kind` — they dispatch straight
        // to `handle_jit_marker__*`. Upstream keys on `op.args[0].value`;
        // pyre keys on the direct_call callee identity since the front-end
        // lowers `driver.jit_merge_point(...)` etc. to `CallTarget::Method`.
        if let Some(key) = jit_marker_key_from_target(target)
            && let Some(ops) = self.try_handle_jit_marker(key, args, graph)
        {
            return RewriteResult::Replace(ops);
        }
        // STRUCTURAL-ADAPTATION (Rust-frontend → RPython rtyper gap).
        // RPython's rtyper resolves `Result`/`Option`-style tagged-union
        // construction to malloc + setfield at lowering time, so by the
        // time a graph reaches jtransform there are no `Ok(_)`-wrapper
        // SpaceOperations left to classify. Pyre's Rust frontend
        // (`front::mir`) lowers every constructor call to `OpKind::Call`
        // uniformly — `Ok(StepResult::Continue)`
        // becomes a residual call to a `(r) → r` funcptr, and the trace
        // recorder emits a `CallR` op for it. The trait-dispatch path
        // executes the same constructor as zero-cost native code and
        // emits NO IR ops, so shadow-walker validation diverges by one
        // synthetic `CallR` at every affected wrapper site. Recognise that known
        // family of transparent Rust wrappers here and elide it via the
        // existing alias mechanism — same orthodoxy as RPython
        // `_noop_rewrite` (jtransform.py:399-401), just at the call-shape
        // surface that pyre still needs because its frontend skips the
        // rtyper step.
        //
        // Identity check is delegated to
        // [`Self::is_synthetic_result_option_ctor`]. The frontend must
        // already have resolved this to `CallTarget::SyntheticTransparentCtor`;
        // jtransform does not perform name-only matching.
        if self.is_synthetic_result_option_ctor(target, args, result_ty) {
            return RewriteResult::Identity(args[0].clone());
        }
        // `rtype_intmask` coerces its input to `lltype.Signed` and returns
        // that value unchanged (`rpython/rtyper/rbuiltin.py:222-225`).  The
        // frontend preserves the coercion as this call-shaped marker, so fold
        // it through the same identity alias used for no-op coercions
        // (`jtransform.py:399-401`).
        if let CallTarget::FunctionPath { segments } = target
            && segments.as_slice() == ["rpython", "rlib", "rarithmetic", "intmask"]
            && args.len() == 1
        {
            return RewriteResult::Identity(args[0].clone());
        }
        // A zero-arg transparent `Tuple` constructor is the unit value `()` —
        // the MIR return-place aggregate of a `-> ()` function (e.g.
        // `w_list_append`).  It carries no payload and no runtime
        // representation, so the residual `Call` to the synthetic ctor would
        // mint a `symbolic_fnaddr` placeholder (absent from
        // `jit_trace_fnaddrs()`) that the codewriter cannot lower — recording
        // it bakes a 64-bit hash as a code address.  Emit a null-ref result
        // placeholder instead; a unit has no fields for anything to read.
        // (A non-empty `Tuple` is a real aggregate and a named unit *enum*
        // variant carries a discriminant — both skip this arm.)
        if let CallTarget::SyntheticTransparentCtor { name, .. } = target
            && name == "Tuple"
            && args.is_empty()
            && matches!(result_ty, ValueType::Ref(_))
        {
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::ConstRefNull,
            }]);
        }
        // RPython `rtyper/rtuple.py:153-169 TupleRepr.newtuple`: every
        // non-empty tuple is a `malloc(GcStruct)` followed by one `setfield`
        // per item.  `front::mir` has already emitted those positional
        // `FieldWrite`s after this synthetic aggregate constructor; lower the
        // constructor itself to the matching allocation instead of treating
        // it as a host call.  A residual synthetic call has no registered
        // function address and would put its stable hash in JitCode as though
        // it were executable code.
        //
        // The frontend names non-empty tuple shapes `Tuple<...>` while the
        // exact bare `Tuple` above is the zero-item `Void` representation.
        // Keep the owner-path gate: the universal MIR tuple aggregate has no
        // Rust nominal owner, whereas a user ADT whose leaf happens to start
        // with `Tuple` is a different allocation policy.
        if let CallTarget::SyntheticTransparentCtor { name, owner_path } = target
            && owner_path.is_empty()
            && majit_ir::descr::is_shaped_tuple_name(name)
            && args.is_empty()
            && let ValueType::Ref(Some(owner)) = result_ty
        {
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::New {
                    owner: owner.clone(),
                },
            }]);
        }
        // RPython `rtyper` lowers a heap-carried `Result` variant to
        // `malloc(GcStruct)` plus its discriminant/payload `setfield`s before
        // `jtransform`; `rewrite_op_malloc` then emits `new(descr)`.
        // Charon exposes the pre-rtyper shape instead: a niladic synthetic
        // ctor followed by a payload `FieldWrite`, and it does not spell the
        // enum tag as a separate MIR operand. Restore the exact low-level
        // shape here. The existing payload write follows this replacement in
        // program order; the tag write is emitted beside the allocation.
        // Recognize the ctor through the front-side rule rather than a second
        // spelling test: it anchors the owner path head at `core::result` and
        // compares the instantiation-stripped leaf for equality, where a
        // `starts_with("Result")` leaf test also accepts an unrelated enum whose
        // name merely begins with it and carries `Ok`/`Err` variants. Both gates
        // decide the same question, so they must not be able to drift.
        if args.is_empty()
            && let Some(is_err) = crate::front::result_exc::result_ctor_kind(target)
            && let ValueType::Ref(Some(owner)) = result_ty
        {
            let name = if is_err { "Err" } else { "Ok" };
            let tag = i64::from(is_err);
            let result_base = owner
                .strip_suffix(format!("::{name}").as_str())
                .unwrap_or(owner)
                .to_string();
            let result = op
                .result
                .clone()
                .expect("SyntheticTransparentCtor must produce a value");
            return RewriteResult::Replace(vec![
                SpaceOperation {
                    result: Some(result.clone()),
                    kind: OpKind::New {
                        owner: owner.clone(),
                    },
                },
                SpaceOperation {
                    result: None,
                    kind: OpKind::FieldWrite {
                        base: result,
                        field: crate::model::FieldDescriptor::new(
                            "__discriminant",
                            Some(result_base),
                        ),
                        value: crate::model::LinkArg::Const(
                            crate::flowspace::model::Constant::new(
                                crate::flowspace::model::ConstValue::Int(tag),
                            ),
                        ),
                        ty: ValueType::Int,
                    },
                },
            ]);
        }
        // `rbuiltin.py:412-418 rtype_const_result` /
        // `translator/rtyper/rbuiltin.rs::rtype_ptr_null`: by the time
        // jtransform runs, `ptr::null[_mut]()` is a typed null pointer
        // constant, not a residual host call. Pyre's rtyper currently types an
        // ephemeral oracle rather than rewriting the surviving model graph,
        // so apply that literal rewrite here. This is the null half of the
        // niche `Option<NonNull<T>>` / `Option<&T>` representation emitted by
        // `front::mir`; leaving it as a call would bake an unregistered
        // symbolic fnaddr into every generated nullity test.
        if let CallTarget::FunctionPath { segments } = target
            && args.is_empty()
            && matches!(result_ty, ValueType::Ref(_))
            && resolves_to_null_ptr_builtin(segments)
        {
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::ConstRefNull,
            }]);
        }
        // `rewrite_op_cast_pointer` → `rewrite_op_same_as`
        // (jtransform.py:254-257): the JIT does not distinguish a
        // down-cast pointer from its source, so the
        // `__cast_pointer/<Root>` marker (front::mir's carrier for the
        // upstream `cast_pointer` op, see `cast_pointer_marker_op`)
        // folds back to the operand alias and emits no jitcode op.
        if let CallTarget::FunctionPath { segments } = target
            && segments.len() == 2
            && segments[0] == "__cast_pointer"
            && args.len() == 1
        {
            return RewriteResult::Identity(args[0].clone());
        }
        // `__pyre_cast_instance/<Root>` — front::mir's pointer-downcast
        // narrow (#298, `mir.rs` emits `Call(["__pyre_cast_instance",
        // root], [v])` for a `Ref → *Struct` reinterpret).  The rtyper
        // lowers it to `cast_pointer` (`rbuiltin.rs rtype_pyre_cast_instance`,
        // `exception_cannot_occur`), which jtransform folds to `same_as`;
        // the charon front-end skips the rtyper, so fold the marker to the
        // operand alias here too.  The JIT does not distinguish a downcast
        // pointer from its source, so this emits no jitcode op.
        // `__pyre_cast_address` — the erasing twin (`p as *mut u8`), which
        // carries no root and so is single-segment.  It reinterprets the
        // same pointer too, and the JIT distinguishes a pointer no more
        // from its erasure than from its downcast, so it folds the same
        // way: the operand alias, no jitcode op.
        if let CallTarget::FunctionPath { segments } = target
            && ((segments.len() == 2 && segments[0] == "__pyre_cast_instance")
                || (segments.len() == 1 && segments[0] == "__pyre_cast_address"))
            && args.len() == 1
        {
            return RewriteResult::Identity(args[0].clone());
        }
        // `<*mut T>::is_null` / `<*const T>::is_null` — the raw-pointer null
        // test.  `front::mir`'s `impl_method_owner` routes it to
        // `CallTarget::Method { name: "is_null", receiver_root }` through the
        // `NON_ADT_OWNER_METHOD_ALLOWLIST` (Charon leaves the primitive `Self`
        // unresolved, so the path stays method-shaped rather than surfacing a
        // panicking `SomeInstance.getattr`).  `ptr_method_is_null`
        // (`unaryop.rs`, shared by `const_ptr`/`mut_ptr` since mutability does
        // not affect the null test) lowers the bound-method to `ptr_iszero`;
        // the charon front-end skips the rtyper, so finish that lowering here
        // the same way the `cast_pointer` family folds above — emit one
        // `ptr_iszero/r>i` over the pointer operand instead of residualising
        // the call to a symbolic helper fnaddr the executor cannot run.
        if let CallTarget::Method {
            name,
            receiver_root: Some(receiver_root),
            ..
        } = target
            && name == "is_null"
            && matches!(receiver_root.as_str(), "mut_ptr" | "const_ptr")
            && args.len() == 1
        {
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::UnaryOp {
                    op: "ptr_iszero".into(),
                    operand: args[0].clone(),
                    result_ty: ValueType::Int,
                },
            }]);
        }
        // `core::ptr::eq(a, b)` — raw-pointer identity comparison.  Like
        // `is_null` above, `front::mir` leaves it as a `FunctionPath` residual
        // because the charon front-end skips the rtyper lowering that turns a
        // pointer `eq` into `ptr_eq` (`jtransform.py:1243-1255` routes `eq`
        // over two Ref operands to `ptr_eq`).  Finish that lowering here: emit
        // a `BinOp("eq")` over the two pointer operands — the assembler maps
        // the `rr` operand shape to `ptr_eq` — instead of residualising the
        // call to a symbolic helper fnaddr the executor cannot run.
        if let CallTarget::FunctionPath { segments } = target
            && segments.len() == 3
            && segments[0] == "core"
            && segments[1] == "ptr"
            && segments[2] == "eq"
            && args.len() == 2
        {
            return RewriteResult::Replace(vec![SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::BinOp {
                    op: "eq".into(),
                    lhs: args[0].clone(),
                    rhs: args[1].clone(),
                    result_ty: ValueType::Int,
                },
            }]);
        }
        // RPython: guess_call_kind(op) → dispatch to handle_*_call
        if let Some(cc) = self.callcontrol.as_mut() {
            let kind = cc.guess_call_kind(op);
            return match kind {
                crate::call::CallKind::Regular => {
                    // RPython call.py:230: the call result must have the same
                    // concretetype as FUNC.RESULT.  In particular, after
                    // exceptiontransform a `Result<(), PyError>` callee has a
                    // void RESULT; emitting `inline_call_r_r` for the front's
                    // aggregate-shaped unit shell would disagree with the
                    // callee's `void_return`.  Reconcile it just like the
                    // residual-call arm below.
                    let effective_result_ty =
                        self.effective_call_result_ty(target, op.result.as_ref(), result_ty);
                    self.handle_regular_call(
                        op,
                        target,
                        args,
                        &effective_result_ty,
                        graph_name,
                        graph,
                    )
                }
                crate::call::CallKind::Residual => {
                    // RPython jtransform.py:456-471:
                    //   calldescr = self.callcontrol.getcalldescr(op, ...)
                    //   op1 = self.rewrite_call(op, 'residual_call', ...)
                    //
                    // RPython ALWAYS produces residual_call_* for residual
                    // calls — the effect is only in the calldescr, NOT in
                    // the opcode name. No dispatch_by_effect.
                    // RPython call.py:220-222: NON_VOID_ARGS + RESULT. Even
                    // for a configured effect override, keep the signature from
                    // getcalldescr() instead of accepting an effect-only descr.
                    let non_void_args = resolve_non_void_arg_types_from_vars(args);
                    // Reconcile a `Result<(), PyError>` scoped callee's
                    // declared void `RESULT` against the `Ref` the front
                    // typed the unit `()` shell (see `effective_call_result_ty`).
                    let effective_result_ty =
                        self.effective_call_result_ty(target, op.result.as_ref(), result_ty);
                    let result_ir_type = self
                        .resolve_call_result(op.result.as_ref(), &effective_result_ty)
                        .ir_type;
                    let cc_ref: &crate::call::CallControl = self.callcontrol.as_deref().unwrap();
                    let extraeffect = classify_call(target, &self.config.call_effects)
                        .map(|(d, _)| d.extra_info.extraeffect);
                    let descriptor = cc_ref.getcalldescr(
                        op,
                        non_void_args,
                        result_ir_type,
                        OopSpecIndex::None,
                        extraeffect,
                        &mut self.analysis_cache,
                        None,
                    );
                    self.handle_residual_call(
                        graph,
                        op,
                        target,
                        descriptor,
                        args,
                        &effective_result_ty,
                        graph_name,
                    )
                }
                crate::call::CallKind::Builtin => {
                    self.handle_builtin_call(op, target, args, result_ty, graph_name, graph)
                }
                crate::call::CallKind::Recursive => {
                    // Raw `result_ty`, no void-Result reconciliation — same
                    // reason as the Regular arm: the inlined-portal callee
                    // carries its unit result as a live `Ref`, not a void.
                    self.handle_recursive_call(op, target, args, result_ty, graph_name, graph)
                }
            };
        }

        // Fallback when no CallControl: effect-only classification (legacy path).
        // RPython: always residual_call_*, effect only in calldescr.
        if let Some((descriptor, _effect)) = classify_call(target, &self.config.call_effects) {
            let non_void_args = resolve_non_void_arg_types_from_vars(args);
            let descriptor = descriptor.with_signature(
                &non_void_args,
                self.resolve_call_result(op.result.as_ref(), result_ty)
                    .ir_type,
            );
            self.handle_residual_call(graph, op, target, descriptor, args, result_ty, graph_name)
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
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
        graph: &mut crate::model::FunctionGraph,
    ) -> RewriteResult {
        // RPython `jtransform.py:484-485`:
        //   oopspec_name, args = support.decode_builtin_call(op)
        //
        // Run the strict-parity decode here so the per-prefix dispatch
        // below sees the normalized (permuted / constant-injected) arg
        // list, not the raw call args.  When the target carries no
        // oopspec registration, `decode_builtin_call` is not invoked —
        // pyre treats that as the unclassified-builtin path and falls
        // through to the `describe_call` / `Keep` branches below.
        //
        // When `oopspec_argnames` is registered for the target,
        // `decode_builtin_call` runs the full `parse_oopspec` +
        // `normalize_opargs` pipeline.  `NormalizedArg::ConstInt(v)`
        // slots are materialised here as fresh `OpKind::ConstInt(v)`
        // ops prepended to whatever the downstream rewrite emits —
        // mirroring upstream's `Constant(obj, ...)` Variable
        // introduction.  Without an argname registry the normalized
        // list is just `Pass(vid)` for each raw arg, so the
        // effective_args == args.to_vec() and no prefix ops are
        // emitted.
        let (decoded_oopspec, effective_args, mut const_prefix_ops) =
            if let Some(cc) = self.callcontrol.as_deref() {
                if cc.get_oopspec(target).is_some() {
                    let (name, normalized) = decode_builtin_call(op, cc);
                    let mut eff: Vec<crate::flowspace::model::Variable> =
                        Vec::with_capacity(normalized.len());
                    let mut prefix: Vec<SpaceOperation> = Vec::new();
                    for slot in normalized {
                        match slot {
                            NormalizedArg::Pass(var) => eff.push(var),
                            // `support.py:723 Constant(obj, lltype.Signed)`.
                            // Only integer literals (`lltype.Signed`-tagged)
                            // reach this arm — `parse_literal_slot` panics
                            // on char literals (`lltype.Char` has no pyre
                            // `ConcreteType` analogue) and routes float
                            // literals to `ConstFloat` below.
                            NormalizedArg::ConstInt(v) => {
                                let var = graph.alloc_value_var_with_type(
                                    crate::codewriter::type_state::ConcreteType::Signed,
                                );
                                prefix.push(SpaceOperation {
                                    result: Some(var.clone()),
                                    kind: OpKind::ConstInt(v),
                                });
                                eff.push(var);
                            }
                            // `support.py:723 Constant(obj, lltype.Float)`
                            NormalizedArg::ConstFloat(bits) => {
                                let var = graph.alloc_value_var_with_type(
                                    crate::codewriter::type_state::ConcreteType::Float,
                                );
                                prefix.push(SpaceOperation {
                                    result: Some(var.clone()),
                                    kind: OpKind::ConstFloat(bits),
                                });
                                eff.push(var);
                            }
                        }
                    }
                    (Some(name), eff, prefix)
                } else {
                    (None, args.to_vec(), Vec::new())
                }
            } else {
                (None, args.to_vec(), Vec::new())
            };
        let args: &[crate::flowspace::model::Variable] = &effective_args;
        let user_oopspec: Option<String> = decoded_oopspec.clone();

        // jtransform.py:487-511 — oopspec dispatch by prefix.
        if let Some(base) = user_oopspec.as_deref() {
            // jtransform.py:497 — jit.* oopspecs → __handle_jit_call
            if base.starts_with("jit.") {
                // `jit.not_in_trace` discards the decoded args (see
                // `_handle_jit_call`), so the ops materialising their
                // constants have no consumer left.
                if base == "jit.not_in_trace" {
                    const_prefix_ops.clear();
                }
                let result =
                    self._handle_jit_call(base, op, target, args, result_ty, graph_name, graph);
                return prepend_const_prefix(&mut const_prefix_ops, result);
            }
            // jtransform.py:488 — list.* / newlist oopspecs → _handle_list_call.
            // Unhandled spellings return `None` and fall through to the
            // residual-call path (RPython raises `NotSupported`).
            if (base.starts_with("list.") || base.starts_with("newlist"))
                && let Some(result) = self._handle_list_call(base, op, args, graph, graph_name)
            {
                return prepend_const_prefix(&mut const_prefix_ops, result);
            }
            // jtransform.py:491-492 — stroruni.* oopspecs → _handle_stroruni_call.
            // Unhandled spellings return `None` and fall through to the
            // residual-call path (RPython raises `NotSupported`).
            if base.starts_with("stroruni.")
                && let Some(result) =
                    self._handle_stroruni_call(base, op, target, args, result_ty, graph_name, graph)
            {
                return prepend_const_prefix(&mut const_prefix_ops, result);
            }
            // jtransform.py:503-504 — rgc.* oopspecs → _handle_rgc_call.
            // Unhandled spellings return `None` and fall through to the
            // residual-call path.
            if base.starts_with("rgc.")
                && let Some(result) =
                    self._handle_rgc_call(base, op, target, args, result_ty, graph_name, graph)
            {
                return prepend_const_prefix(&mut const_prefix_ops, result);
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
        //
        // jtransform.py:2186 _handle_dict_lookup_call passes
        // `extradescr=[cpu.fielddescrof(STRUCT, 'entries'), cpu.arraydescrof(STRUCT.entries.TO)]`
        // derived from `op.args[1].concretetype.TO`. pyre's
        // `FunctionGraph::concretetype_of(&v)` collapses lltype to four kinds, so
        // the dict struct is not recoverable here — extradescrs stays
        // None until full lltype propagation lands.
        // OptHeap::_optimize_call_dict_lookup returns false on None extradescrs
        // and the call falls through emit_residual_call (heap.py:472-475 emit
        // → force_from_effectinfo on the call's own effectinfo).
        let non_void_args = resolve_non_void_arg_types_from_vars(args);
        let result_ir_type = self
            .resolve_call_result(op.result.as_ref(), result_ty)
            .ir_type;
        let descriptor = {
            let cc_ref: &crate::call::CallControl = self.callcontrol.as_deref().unwrap();
            cc_ref.getcalldescr(
                op,
                non_void_args,
                result_ir_type,
                oopspecindex,
                extraeffect_override,
                &mut self.analysis_cache,
                None,
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
        if oopspecindex != OopSpecIndex::None
            && let Some(cc) = self.callcontrol.as_mut()
        {
            let func_as_int = cc.fnaddr_for_target(target) as u64;

            cc.callinfocollection
                .add(oopspecindex, descriptor.to_descr_ref(), func_as_int);
            cc.callinfocollection
                .register_func_name(func_as_int, format!("{target}"));
        }

        // RPython jtransform.py:2003-2007: __handle_oopspec_call always
        // produces residual_call_*, appends -live- if calldescr_canraise.
        // Effect is only in the calldescr, never in the opcode name.
        let result =
            self.handle_residual_call(graph, op, target, descriptor, args, result_ty, graph_name);
        prepend_const_prefix(&mut const_prefix_ops, result)
    }

    /// Port of `jtransform.py:2049 _handle_stroruni_call`.
    ///
    /// `SoU = args[0].concretetype` classifies the operand as `Ptr(STR)` /
    /// `Ptr(UNICODE)` / `Ptr(BYTEARRAY)` (jtransform.py:2058-2077); pyre reads
    /// it from the flowspace `Variable::concretetype()` via
    /// [`stroruni_first_arg_kind`]. `Ptr(BYTEARRAY)` is `raise NotSupported`
    /// (jtransform.py:2074) — pyre returns `None`, falling through to the
    /// residual-call path — and any other shape is `else: assert 0`
    /// (jtransform.py:2077).
    ///
    /// - `copy_contents` (jtransform.py:2079-2086) lowers to a
    ///   `copystrcontent` / `copyunicodecontent` [`OpKind::LoweredBlackholeOp`],
    ///   the same representation the Spine-B string transducer produces, so
    ///   flatten / assembly already lower it to the matching jitcode bytecode.
    /// - `concat` / `slice` / `cmp` / `copy_string_to_raw`
    ///   (jtransform.py:2059-2072, 2124-2128) map to the `OS_STR_*` / `OS_UNI_*`
    ///   oopspecindex and lower to a residual call. `concat` / `slice` can
    ///   raise `MemoryError` (`EF_ELIDABLE_OR_MEMORYERROR`), `cmp` /
    ///   `copy_string_to_raw` cannot (`EF_ELIDABLE_CANNOT_RAISE`).
    /// - `equal` (jtransform.py:2087-2122) additionally registers the
    ///   `OS_STREQ_*` / `OS_UNIEQ_*` slice-comparison helper variants via
    ///   `_register_extra_helper`, which pyre has not ported; that spelling
    ///   returns `None` and falls through to the residual-call path until the
    ///   helper-registration machinery lands.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    fn _handle_stroruni_call(
        &mut self,
        oopspec_name: &str,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
        graph: &mut FunctionGraph,
    ) -> Option<RewriteResult> {
        // SoU = args[0].concretetype (jtransform.py:2050): the STR vs UNICODE
        // spelling selects the OS_STR_* vs OS_UNI_* oopspecindex family below.
        let kind = stroruni_first_arg_kind(args.first()?);

        // jtransform.py:2079-2086 — copy_contents lowers to a blackhole op.
        if oopspec_name == "stroruni.copy_contents" {
            let opname = match kind {
                StrOrUniKind::Str => "copystrcontent",
                StrOrUniKind::Unicode => "copyunicodecontent",
                // BYTEARRAY → NotSupported (jtransform.py:2074): fall through
                // to the residual-call path.
                StrOrUniKind::ByteArray => return None,
                // jtransform.py:2077 `else: assert 0, "args[0].concretetype
                // must be STR or UNICODE"`. A `stroruni.copy_contents` oopspec
                // guarantees a string pointer, so any other shape is a
                // should-never-happen invariant violation.
                StrOrUniKind::Other => {
                    panic!("args[0].concretetype must be STR or UNICODE")
                }
            };
            return Some(RewriteResult::Replace(vec![SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::LoweredBlackholeOp {
                    opname: opname.to_string(),
                    args: args.to_vec(),
                },
            }]));
        }

        // jtransform.py:2087-2122 — before falling through to the common
        // residual-call tail, stroruni.equal registers seven OS_STREQ_* /
        // OS_UNIEQ_* slice-comparison side-helpers (str.eq_slice_checknull /
        // eq_slice_nonnull / eq_slice_char / eq_nonnull / eq_nonnull_char /
        // eq_checknull_char / eq_lengthok, each shifted by _OS_offset_uni for
        // UNICODE) via _register_extra_helper. That registration is blocked:
        // _register_extra_helper → support::builtin_func_for_spec →
        // setup_extra_builtin resolves each helper's host fnaddr and panics on
        // a miss, and pyre deliberately does not host-bind the str.eq_*
        // helpers — Python strings are W_UnicodeObject (native WTF-8), not the
        // rstr.STR `{hash, chars}` GcStruct these helpers index (see
        // jit_fnaddr.rs `jit_trace_fnaddrs` and blackhole.rs next to
        // `bhimpl_int_and`). The side-registrations are consumed only by the
        // string-optimization pass; their absence does not affect the main
        // OS_STR_EQUAL / OS_UNI_EQUAL residual call emitted by the tail below,
        // which resolves the primary helper through fnaddr_for_target like
        // concat/slice/cmp. Wire the extra-helper loop once the rstr.STR
        // runtime layout and str.eq_* host bodies land.

        // jtransform.py:2059-2072 — the OS_STR_* / OS_UNI_* index selected by
        // the STR vs UNICODE operand. BYTEARRAY → NotSupported (None); any
        // other shape → assert 0 (jtransform.py:2074-2077).
        let oopspecindex = match (oopspec_name, kind) {
            ("stroruni.concat", StrOrUniKind::Str) => OopSpecIndex::StrConcat,
            ("stroruni.concat", StrOrUniKind::Unicode) => OopSpecIndex::UniConcat,
            ("stroruni.slice", StrOrUniKind::Str) => OopSpecIndex::StrSlice,
            ("stroruni.slice", StrOrUniKind::Unicode) => OopSpecIndex::UniSlice,
            ("stroruni.equal", StrOrUniKind::Str) => OopSpecIndex::StrEqual,
            ("stroruni.equal", StrOrUniKind::Unicode) => OopSpecIndex::UniEqual,
            ("stroruni.cmp", StrOrUniKind::Str) => OopSpecIndex::StrCmp,
            ("stroruni.cmp", StrOrUniKind::Unicode) => OopSpecIndex::UniCmp,
            ("stroruni.copy_string_to_raw", StrOrUniKind::Str) => OopSpecIndex::StrCopyToRaw,
            ("stroruni.copy_string_to_raw", StrOrUniKind::Unicode) => OopSpecIndex::UniCopyToRaw,
            // BYTEARRAY → raise NotSupported (jtransform.py:2074).
            (_, StrOrUniKind::ByteArray) => return None,
            // else: assert 0 (jtransform.py:2077).
            (_, StrOrUniKind::Other) => {
                panic!("args[0].concretetype must be STR or UNICODE")
            }
            // An unported / unknown stroruni.* spelling: fall through to the
            // residual-call path.
            _ => return None,
        };

        // jtransform.py:2051-2057, 2124-2128 — concat/slice can raise
        // MemoryError; cmp/copy_string_to_raw cannot.
        let extra = match oopspec_name {
            "stroruni.concat" | "stroruni.slice" => {
                majit_ir::descr::ExtraEffect::ElidableOrMemoryError
            }
            _ => majit_ir::descr::ExtraEffect::ElidableCannotRaise,
        };
        Some(self._handle_oopspec_call(
            graph,
            op,
            target,
            args,
            result_ty,
            graph_name,
            oopspecindex,
            Some(extra),
            None,
        ))
    }

    /// Port of `jtransform.py:2193 _handle_rgc_call`, `ll_shrink_array` arm:
    ///
    /// ```python
    /// def _handle_rgc_call(self, op, oopspec_name, args):
    ///     if oopspec_name == 'rgc.ll_shrink_array':
    ///         return self._handle_oopspec_call(op, args,
    ///                     EffectInfo.OS_SHRINK_ARRAY, EffectInfo.EF_CAN_RAISE)
    /// ```
    ///
    /// Lowers to a `residual_call` whose calldescr carries the `ShrinkArray`
    /// oopspecindex and the `CanRaise` effect, so `handle_residual_call`
    /// appends the trailing `-live-`.
    ///
    /// jtransform.py:2197 `raise NotImplementedError(oopspec_name)` for any
    /// other `rgc.*` spelling. Pyre instead returns `None` so the call falls
    /// through to the residual-call path — the same "unported oopspec →
    /// residual" convention the `list.*` / `stroruni.*` sibling handlers use,
    /// which keeps rgc oopspecs pyre has not lowered yet (e.g.
    /// `rgc.ll_arraycopy`) as ordinary residual calls rather than aborting
    /// codewriting.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    fn _handle_rgc_call(
        &mut self,
        oopspec_name: &str,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
        graph: &mut FunctionGraph,
    ) -> Option<RewriteResult> {
        match oopspec_name {
            "rgc.ll_shrink_array" => Some(self._handle_oopspec_call(
                graph,
                op,
                target,
                args,
                result_ty,
                graph_name,
                OopSpecIndex::ShrinkArray,
                Some(majit_ir::descr::ExtraEffect::CanRaise),
                None,
            )),
            _ => None,
        }
    }

    /// Port of `jtransform.py:1762 _handle_list_call` for pyre's
    /// strategy-tagged Integer-storage list oopspec leaves
    /// (`ll_list_int_{length,getitem_fast,setitem_fast}` in
    /// `listobject.rs`, annotated
    /// `#[oopspec("list.int_{len,getitem,setitem}")]`).
    ///
    /// Lowers the oopspec call into the typed field / array ops the
    /// heapcache understands instead of an opaque residual call.  pyre's
    /// Integer storage is a nested inline `IntArray`
    /// (`W_ListObject.int_items: IntArray { block: Ptr(GcArray(Signed)),
    /// ptr, len, .. }`), so the items live behind the `int_items.block`
    /// GC array:
    ///   `list.int_len(l)`         → getfield_gc_i(l, int_items.len)
    ///   `list.int_getitem(l, i)`  → getfield_gc_r(l, int_items.block);
    ///                                getarrayitem_gc_i(block, i)
    ///   `list.int_setitem(l,i,v)` → getfield_gc_r(l, int_items.block);
    ///                                setarrayitem_gc(block, i, v)
    ///
    /// pyre splits the fused `getlistitem_gc` / `setlistitem_gc` resops
    /// (`do_resizable_list_getitem/setitem`, jtransform.py:1954-1972)
    /// into the explicit getfield(block) + get/setarrayitem pair because
    /// the runtime exposes the backing GC array through the
    /// `int_items.block` field (`Ptr(GcArray(Signed))`) rather than a
    /// fused interior descr.
    ///
    /// The `_fast` leaves carry a non-negative index by contract
    /// (`jtransform.py:1799 _get_list_nonneg_canraise_flags` → no
    /// `check_neg_index`), so the index operand is used directly.
    ///
    /// Returns `None` for any oopspec spelling this does not handle, so
    /// the caller falls through to the residual path
    /// (`jtransform.py:1796` raises `NotSupported`).
    fn _handle_list_call(
        &mut self,
        oopspec_name: &str,
        op: &SpaceOperation,
        args: &[crate::flowspace::model::Variable],
        graph: &mut crate::model::FunctionGraph,
        graph_name: &str,
    ) -> Option<RewriteResult> {
        use crate::codewriter::type_state::ConcreteType;
        // Field owner for the `W_ListObject` storage struct.  The dotted
        // names address the fused offsets the runtime descr group
        // exposes (`int_items.len` → `list_int_items_len_descr`,
        // `int_items.block` → `list_int_items_block_descr`); the
        // descr-resolution layer maps them to the matching
        // `W_LIST_DESCR_GROUP` entries.
        const LIST_OWNER: &str = "W_ListObject";

        let (detail, ops): (&str, Vec<SpaceOperation>) = match oopspec_name {
            "list.int_len" => {
                let l = args.first()?.clone();
                (
                    "list.int_len → getfield_gc_i(int_items.len)",
                    vec![SpaceOperation {
                        result: op.result.clone(),
                        kind: OpKind::FieldRead {
                            base: l,
                            field: FieldDescriptor::new(
                                "int_items.len",
                                Some(LIST_OWNER.to_string()),
                            ),
                            ty: ValueType::Int,
                            pure: false,
                        },
                    }],
                )
            }
            "list.int_getitem" => {
                let l = args.first()?.clone();
                let index = args.get(1)?.clone();
                let block = graph.alloc_value_var_with_type(ConcreteType::GcRef);
                (
                    "list.int_getitem → getfield_gc_r(int_items.block) + getarrayitem_gc_i",
                    vec![
                        SpaceOperation {
                            result: Some(block.clone()),
                            kind: OpKind::FieldRead {
                                base: l,
                                field: FieldDescriptor::new(
                                    "int_items.block",
                                    Some(LIST_OWNER.to_string()),
                                ),
                                ty: ValueType::Ref(None),
                                pure: false,
                            },
                        },
                        SpaceOperation {
                            result: op.result.clone(),
                            kind: OpKind::ArrayRead {
                                base: block,
                                index,
                                item_ty: ValueType::Int,
                                array_type_id: None,
                                nolength: false,
                                pure: false,
                            },
                        },
                    ],
                )
            }
            // `ll_getitem_foldable_nonneg` (rlist.py:721-724, `oopspec =
            // 'list.getitem_foldable(l, index)'`) — selected by
            // `rtype_getitem` when `not listdef.listitem.mutated`
            // (rlist.py:256-258).  Same block-then-element decomposition
            // as `list.int_getitem`, except the element load is the
            // foldable `getarrayitem_gc_i_pure` (`pure: true`).  The
            // `int_items.block` FieldRead stays `pure: false` — only the
            // element load is foldable; the backing block pointer may
            // still move under a resize.
            "list.int_getitem_foldable" => {
                let l = args.first()?.clone();
                let index = args.get(1)?.clone();
                let block = graph.alloc_value_var_with_type(ConcreteType::GcRef);
                (
                    "list.int_getitem_foldable → getfield_gc_r(int_items.block) + getarrayitem_gc_i_pure",
                    vec![
                        SpaceOperation {
                            result: Some(block.clone()),
                            kind: OpKind::FieldRead {
                                base: l,
                                field: FieldDescriptor::new(
                                    "int_items.block",
                                    Some(LIST_OWNER.to_string()),
                                ),
                                ty: ValueType::Ref(None),
                                pure: false,
                            },
                        },
                        SpaceOperation {
                            result: op.result.clone(),
                            kind: OpKind::ArrayRead {
                                base: block,
                                index,
                                item_ty: ValueType::Int,
                                array_type_id: None,
                                nolength: false,
                                pure: true,
                            },
                        },
                    ],
                )
            }
            "list.int_setitem" => {
                let l = args.first()?.clone();
                let index = args.get(1)?.clone();
                let value = args.get(2)?.clone();
                let block = graph.alloc_value_var_with_type(ConcreteType::GcRef);
                (
                    "list.int_setitem → getfield_gc_r(int_items.block) + setarrayitem_gc",
                    vec![
                        SpaceOperation {
                            result: Some(block.clone()),
                            kind: OpKind::FieldRead {
                                base: l,
                                field: FieldDescriptor::new(
                                    "int_items.block",
                                    Some(LIST_OWNER.to_string()),
                                ),
                                ty: ValueType::Ref(None),
                                pure: false,
                            },
                        },
                        SpaceOperation {
                            result: op.result.clone(),
                            kind: OpKind::ArrayWrite {
                                base: block,
                                index,
                                value: crate::model::LinkArg::Value(value),
                                item_ty: ValueType::Int,
                                array_type_id: None,
                                nolength: false,
                            },
                        },
                    ],
                )
            }
            "list.int_capacity" => {
                // Capacity is `len(l.items)` (rlist.py:251) — the backing
                // block's offset-0 length header, not a direct field of the
                // list. Load `int_items.block` then read `ItemsBlock.capacity`
                // off it, mirroring `list.obj_capacity`. The capacity descr is
                // a block-kind-agnostic offset-0 scalar read, so it resolves
                // the TypedItemsBlock (int storage) header the same way.
                let l = args.first()?.clone();
                let block = graph.alloc_value_var_with_type(ConcreteType::GcRef);
                (
                    "list.int_capacity → getfield_gc_r(int_items.block) + getfield_gc_i(block.capacity)",
                    vec![
                        SpaceOperation {
                            result: Some(block.clone()),
                            kind: OpKind::FieldRead {
                                base: l,
                                field: FieldDescriptor::new(
                                    "int_items.block",
                                    Some(LIST_OWNER.to_string()),
                                ),
                                ty: ValueType::Ref(None),
                                pure: false,
                            },
                        },
                        SpaceOperation {
                            result: op.result.clone(),
                            kind: OpKind::FieldRead {
                                base: block,
                                field: FieldDescriptor::new(
                                    "capacity",
                                    Some("ItemsBlock".to_string()),
                                ),
                                ty: ValueType::Int,
                                pure: false,
                            },
                        },
                    ],
                )
            }
            "list.int_set_len" => {
                let l = args.first()?.clone();
                let n = args.get(1)?.clone();
                (
                    "list.int_set_len → setfield_gc_i(int_items.len)",
                    vec![SpaceOperation {
                        result: op.result.clone(),
                        kind: OpKind::FieldWrite {
                            base: l,
                            field: FieldDescriptor::new(
                                "int_items.len",
                                Some(LIST_OWNER.to_string()),
                            ),
                            value: crate::model::LinkArg::Value(n),
                            ty: ValueType::Int,
                        },
                    }],
                )
            }
            // Float-strategy storage leaves, mirroring the Integer leaves
            // but addressing `float_items.{len,block}` and holding
            // unboxed `f64` scalars — the element store lowers to
            // `setarrayitem_gc_f` (item type Float, no write barrier).
            "list.float_len" => {
                let l = args.first()?.clone();
                (
                    "list.float_len → getfield_gc_i(float_items.len)",
                    vec![SpaceOperation {
                        result: op.result.clone(),
                        kind: OpKind::FieldRead {
                            base: l,
                            field: FieldDescriptor::new(
                                "float_items.len",
                                Some(LIST_OWNER.to_string()),
                            ),
                            ty: ValueType::Int,
                            pure: false,
                        },
                    }],
                )
            }
            "list.float_setitem" => {
                let l = args.first()?.clone();
                let index = args.get(1)?.clone();
                let value = args.get(2)?.clone();
                let block = graph.alloc_value_var_with_type(ConcreteType::GcRef);
                (
                    "list.float_setitem → getfield_gc_r(float_items.block) + setarrayitem_gc_f",
                    vec![
                        SpaceOperation {
                            result: Some(block.clone()),
                            kind: OpKind::FieldRead {
                                base: l,
                                field: FieldDescriptor::new(
                                    "float_items.block",
                                    Some(LIST_OWNER.to_string()),
                                ),
                                ty: ValueType::Ref(None),
                                pure: false,
                            },
                        },
                        SpaceOperation {
                            result: op.result.clone(),
                            kind: OpKind::ArrayWrite {
                                base: block,
                                index,
                                value: crate::model::LinkArg::Value(value),
                                item_ty: ValueType::Float,
                                array_type_id: None,
                                nolength: false,
                            },
                        },
                    ],
                )
            }
            "list.float_capacity" => {
                // Capacity is `len(l.items)` (rlist.py:251) — the backing
                // block's offset-0 length header, not a direct field of the
                // list. Load `float_items.block` then read `ItemsBlock.capacity`
                // off it, mirroring `list.int_capacity`/`list.obj_capacity`. The
                // capacity descr is a block-kind-agnostic offset-0 scalar read,
                // so it resolves the TypedItemsBlock (float storage) header the
                // same way.
                let l = args.first()?.clone();
                let block = graph.alloc_value_var_with_type(ConcreteType::GcRef);
                (
                    "list.float_capacity → getfield_gc_r(float_items.block) + getfield_gc_i(block.capacity)",
                    vec![
                        SpaceOperation {
                            result: Some(block.clone()),
                            kind: OpKind::FieldRead {
                                base: l,
                                field: FieldDescriptor::new(
                                    "float_items.block",
                                    Some(LIST_OWNER.to_string()),
                                ),
                                ty: ValueType::Ref(None),
                                pure: false,
                            },
                        },
                        SpaceOperation {
                            result: op.result.clone(),
                            kind: OpKind::FieldRead {
                                base: block,
                                field: FieldDescriptor::new(
                                    "capacity",
                                    Some("ItemsBlock".to_string()),
                                ),
                                ty: ValueType::Int,
                                pure: false,
                            },
                        },
                    ],
                )
            }
            "list.float_set_len" => {
                let l = args.first()?.clone();
                let n = args.get(1)?.clone();
                (
                    "list.float_set_len → setfield_gc_i(float_items.len)",
                    vec![SpaceOperation {
                        result: op.result.clone(),
                        kind: OpKind::FieldWrite {
                            base: l,
                            field: FieldDescriptor::new(
                                "float_items.len",
                                Some(LIST_OWNER.to_string()),
                            ),
                            value: crate::model::LinkArg::Value(n),
                            ty: ValueType::Int,
                        },
                    }],
                )
            }
            // Object-strategy storage leaves. The live length is the
            // `W_ListObject.length` header; the items live behind the
            // `items` GcArray block (`Ptr(GcArray(OBJECTPTR))`) whose
            // offset-0 length header IS the allocated capacity. Element
            // stores are GC-ref writes, so the array store lowers to
            // `setarrayitem_gc_r` (write barrier carried by the resop).
            "list.obj_len" => {
                let l = args.first()?.clone();
                (
                    "list.obj_len → getfield_gc_i(length)",
                    vec![SpaceOperation {
                        result: op.result.clone(),
                        kind: OpKind::FieldRead {
                            base: l,
                            field: FieldDescriptor::new("length", Some(LIST_OWNER.to_string())),
                            ty: ValueType::Int,
                            pure: false,
                        },
                    }],
                )
            }
            "list.obj_capacity" => {
                let l = args.first()?.clone();
                let block = graph.alloc_value_var_with_type(ConcreteType::GcRef);
                (
                    "list.obj_capacity → getfield_gc_r(items) + arraylen_gc(block)",
                    vec![
                        SpaceOperation {
                            result: Some(block.clone()),
                            kind: OpKind::FieldRead {
                                base: l,
                                field: FieldDescriptor::new("items", Some(LIST_OWNER.to_string())),
                                ty: ValueType::Ref(None),
                                pure: false,
                            },
                        },
                        SpaceOperation {
                            result: op.result.clone(),
                            kind: OpKind::ArrayLen {
                                base: block,
                                array_type_id: None,
                                nolength: false,
                            },
                        },
                    ],
                )
            }
            "list.obj_set_len" => {
                let l = args.first()?.clone();
                let n = args.get(1)?.clone();
                (
                    "list.obj_set_len → setfield_gc_i(length)",
                    vec![SpaceOperation {
                        result: op.result.clone(),
                        kind: OpKind::FieldWrite {
                            base: l,
                            field: FieldDescriptor::new("length", Some(LIST_OWNER.to_string())),
                            value: crate::model::LinkArg::Value(n),
                            ty: ValueType::Int,
                        },
                    }],
                )
            }
            // `do_fixed_newlist_clear` (jtransform.py:1858-1863):
            //   v_length = self._get_initial_newlist_length(op, args)
            //   return SpaceOperation('new_array_clear', [v_length, arraydescr],
            //                         op.result)
            // `_ll_alloc_and_clear` (rlist.py:522-528, `@jit.oopspec(
            // "newlist_clear(count)")`) reaches here through
            // `_handle_list_call`; the `count` arg is the initial length.
            // `_get_initial_newlist_length` (jtransform.py:1835-1842):
            // `args[0]` when present, else the constant 0 — pyre carries the
            // materialised `count` Variable, so a `newlist_clear(count)`
            // callsite always supplies `args[0]`.  The cleared list holds GC
            // pointers, so the arraydescr's item type is `Ref` — the exact
            // shape that makes `do_fixed_newlist` select `new_array_clear`
            // over `new_array` (jtransform.py:1851-1855).
            "newlist_clear" => {
                let length = args.first()?.clone();
                // Fork on the result's list layout, mirroring upstream's
                // `LIST = op.result.concretetype.TO; resizable =
                // isinstance(LIST, lltype.GcStruct)` split in
                // `_handle_list_call` (jtransform.py:1762-1785):
                // `do_resizable_newlist_clear` (jtransform.py:1938) allocates
                // the `"list"` header struct plus a cleared items array,
                // `do_fixed_newlist_clear` (jtransform.py:1858) emits a bare
                // cleared array.  `item_ty` / `array_type_id` are recovered
                // from the result element lltype in BOTH cases — never
                // hardcoded — so a non-pointer element list is not GC-traced
                // as a pointer array, and a resized result carries the struct
                // header instead of a bare array (C1).
                let (detail, kind) = match newlist_clear_shape(op.result.as_ref()) {
                    NewlistClearShape::Resized {
                        item_ty,
                        array_type_id,
                    } => (
                        "newlist_clear → newlist_clear(count, arraydescr) \
                         [resized list header]",
                        OpKind::NewListClear {
                            length,
                            item_ty,
                            array_type_id,
                        },
                    ),
                    NewlistClearShape::Fixed {
                        item_ty,
                        array_type_id,
                    } => (
                        "newlist_clear → new_array_clear(count, arraydescr) \
                         [fixed array]",
                        OpKind::NewArrayClear {
                            length,
                            item_ty,
                            array_type_id,
                        },
                    ),
                    // OBJECTPTR / None fallback: the pre-fork behavior, kept
                    // until N7 rewrites `_handle_list_call`'s tests with typed
                    // results.  A generic `OBJECTPTR` result (e.g. a test that
                    // stamped `alloc_value_var_with_type(GcRef)`) or a missing
                    // concretetype hits this arm.
                    NewlistClearShape::Fallback => (
                        "newlist_clear → new_array_clear(count, arraydescr)",
                        OpKind::NewArrayClear {
                            length,
                            item_ty: ValueType::Ref(None),
                            array_type_id: None,
                        },
                    ),
                };
                (
                    detail,
                    vec![SpaceOperation {
                        result: op.result.clone(),
                        kind,
                    }],
                )
            }
            "list.obj_setitem" => {
                let l = args.first()?.clone();
                let index = args.get(1)?.clone();
                let value = args.get(2)?.clone();
                let block = graph.alloc_value_var_with_type(ConcreteType::GcRef);
                (
                    "list.obj_setitem → getfield_gc_r(items) + setarrayitem_gc_r",
                    vec![
                        SpaceOperation {
                            result: Some(block.clone()),
                            kind: OpKind::FieldRead {
                                base: l,
                                field: FieldDescriptor::new("items", Some(LIST_OWNER.to_string())),
                                ty: ValueType::Ref(None),
                                pure: false,
                            },
                        },
                        SpaceOperation {
                            result: op.result.clone(),
                            kind: OpKind::ArrayWrite {
                                base: block,
                                index,
                                value: crate::model::LinkArg::Value(value),
                                item_ty: ValueType::Ref(None),
                                array_type_id: None,
                                nolength: false,
                            },
                        },
                    ],
                )
            }
            _ => return None,
        };
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: detail.to_string(),
        });
        Some(RewriteResult::Replace(ops))
    }

    /// RPython: `Transformer.handle_regular_call(op)`.
    /// Callee is a candidate graph — emit `inline_call_*` referencing
    /// the callee's JitCode. The meta-interpreter will descend into
    /// the callee JitCode at runtime.
    ///
    /// RPython jtransform.py:473-482.
    #[expect(
        clippy::arc_with_non_send_sync,
        reason = "Arc preserves shared runtime descriptor/JitCode identity while non-Send translator payload remains confined to the single-threaded build phase"
    )]
    fn handle_regular_call(
        &mut self,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
        graph: &mut crate::model::FunctionGraph,
    ) -> RewriteResult {
        // RPython jtransform.py:477-478: get_jitcode(targetgraph)
        //
        // Route through the CallControl-aware qualified path so inherent
        // methods like `PyFrame::pop_top` resolve to their
        // `function_graphs[["PyFrame", "pop_top"]]` entry. The stateless
        // `target_to_call_path` fallback only yields the bare method name,
        // which never matches `CallControl::register_inherent_method`'s
        // qualified key (`call.rs:941`) and leaves the shell body-less in
        // `drain_pending_graphs`.
        let jitcode = if let Some(cc) = self.callcontrol.as_mut() {
            let path = cc
                .target_to_path(target)
                .unwrap_or_else(|| target_to_call_path(target));
            crate::jitcode::JitCodeHandle::new(cc.get_jitcode(&path))
        } else {
            crate::jitcode::JitCodeHandle::new(std::sync::Arc::new(crate::jitcode::JitCode::new(
                "<missing-callcontrol>",
            )))
        };
        // RPython jtransform.py:480: rewrite_call(op, 'inline_call', [jitcode])
        // Split args by kind (RPython make_three_lists)
        let (args_i, args_r, args_f) = self.rewrite_call_three_lists(args, graph_name);
        let result_kind = self.resolve_call_result(op.result.as_ref(), result_ty).kind;
        self.stamp_value_kind_from_value_type(graph, op.result.clone(), result_ty);

        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("call {target} → inline_call[jitcode={:?}]", jitcode.name),
        });
        self.calls_classified += 1;
        // RPython jtransform.py:480-481: inline_call always followed by -live-
        RewriteResult::Replace(vec![
            SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::InlineCall {
                    jitcode,
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
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
        graph: &mut crate::model::FunctionGraph,
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
        let (greens_i, greens_r, greens_f) = self.make_three_lists_from_vars(green_args);
        let (reds_i, reds_r, reds_f) = self.make_three_lists_from_vars(red_args);
        let result_kind = self.resolve_call_result(op.result.as_ref(), result_ty).kind;
        self.stamp_value_kind_from_value_type(graph, op.result.clone(), result_ty);

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
            result: op.result.clone(),
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
    fn promote_greens(
        &self,
        green_args: &[crate::flowspace::model::Variable],
    ) -> Vec<SpaceOperation> {
        let mut ops = Vec::new();
        for var in green_args {
            let kind = self.get_value_kind_var(var);
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
                    value: var.clone(),
                    kind_char: kind,
                },
            });
        }
        ops
    }

    /// RPython: `Transformer.__handle_jit_call(op, oopspec_name, args)` (jtransform.py:1730-1757).
    /// Dispatches jit.* oopspec calls to dedicated opcodes or __handle_oopspec_call.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    fn _handle_jit_call(
        &mut self,
        oopspec_name: &str,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
        graph: &mut crate::model::FunctionGraph,
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
                let value_var = args[0].clone();
                let kind_char = self.get_value_kind_var(&value_var);
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!("jit.assert_green → {kind_char}_assert_green"),
                });
                RewriteResult::Replace(vec![SpaceOperation {
                    result: None,
                    kind: OpKind::AssertGreen {
                        value: value_var,
                        kind_char,
                    },
                }])
            }
            // jtransform.py:1736-1737
            "jit.current_trace_length" => {
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: "jit.current_trace_length → current_trace_length".to_string(),
                });
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::CurrentTraceLength,
                }])
            }
            // jtransform.py:1738-1740
            "jit.isconstant" => {
                let value_var = args[0].clone();
                let kind_char = self.get_value_kind_var(&value_var);
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!("jit.isconstant → {kind_char}_isconstant"),
                });
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::IsConstant {
                        value: value_var,
                        kind_char,
                    },
                }])
            }
            // jtransform.py:1741-1743
            "jit.isvirtual" => {
                let value_var = args[0].clone();
                let kind_char = self.get_value_kind_var(&value_var);
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!("jit.isvirtual → {kind_char}_isvirtual"),
                });
                RewriteResult::Replace(vec![SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::IsVirtual {
                        value: value_var,
                        kind_char,
                    },
                }])
            }
            // jtransform.py:1744-1747
            "jit.force_virtual" => self._handle_oopspec_call(
                graph,
                op,
                target,
                args,
                result_ty,
                graph_name,
                OopSpecIndex::JitForceVirtual,
                Some(majit_ir::descr::ExtraEffect::ForcesVirtualOrVirtualizable),
                None,
            ),
            // jtransform.py:1748-1755
            "jit.not_in_trace" => {
                // jtransform.py:1750-1753: not_in_trace must return void
                assert!(
                    *result_ty == ValueType::Void,
                    "jit.not_in_trace() function must return None"
                );
                // jtransform.py:1749 "ignore 'args' and use the original
                // 'op.args'": the residual call carries the arguments the
                // call site actually passes, not the oopspec-decoded list.
                let original_args = match &op.kind {
                    OpKind::Call { args, .. } => args.clone(),
                    _ => unreachable!("_handle_jit_call reached on non-Call op"),
                };
                self._handle_oopspec_call(
                    graph,
                    op,
                    target,
                    &original_args,
                    result_ty,
                    graph_name,
                    OopSpecIndex::NotInTrace,
                    None,
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
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    fn _handle_oopspec_call(
        &mut self,
        graph: &mut FunctionGraph,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
        oopspecindex: OopSpecIndex,
        extraeffect: Option<majit_ir::descr::ExtraEffect>,
        extradescrs: Option<Vec<majit_ir::DescrRef>>,
    ) -> RewriteResult {
        // jtransform.py:1990-1993
        let non_void_args = resolve_non_void_arg_types_from_vars(args);
        let result_ir_type = self
            .resolve_call_result(op.result.as_ref(), result_ty)
            .ir_type;
        let descriptor = {
            let cc_ref: &crate::call::CallControl = self.callcontrol.as_deref().unwrap();
            cc_ref.getcalldescr(
                op,
                non_void_args,
                result_ir_type,
                oopspecindex,
                extraeffect,
                &mut self.analysis_cache,
                extradescrs,
            )
        };
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("oopspec {oopspecindex:?} → residual_call"),
        });
        self.calls_classified += 1;
        // jtransform.py:1999-2002: callinfocollection.add
        if oopspecindex != OopSpecIndex::None
            && let Some(cc) = self.callcontrol.as_mut()
        {
            let func_as_int = cc.fnaddr_for_target(target) as u64;
            cc.callinfocollection
                .add(oopspecindex, descriptor.to_descr_ref(), func_as_int);
            cc.callinfocollection
                .register_func_name(func_as_int, format!("{target}"));
        }
        // jtransform.py:2003-2008: residual_call + optional -live-
        self.handle_residual_call(graph, op, target, descriptor, args, result_ty, graph_name)
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
    // is deliberate. Keeping the body
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
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    fn _rewrite_op_cond_call(
        &mut self,
        graph: &mut FunctionGraph,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
        is_value: bool,
    ) -> RewriteResult {
        // jtransform.py:1666-1672: validate no floats, ≤4+2 args
        for arg in args {
            if self.get_value_kind_var(arg) == 'f' {
                panic!("Conditional call does not support floats");
            }
        }
        if args.len() > 4 + 2 {
            panic!("Conditional call does not support more than 4 arguments");
        }
        // jtransform.py:1673-1676: calldescr from function call (args[1:] → result)
        let condition_or_value_var = args[0].clone();
        let func_args: &[crate::flowspace::model::Variable] =
            if args.len() > 1 { &args[1..] } else { &[] };
        let non_void_args = resolve_non_void_arg_types_from_vars(func_args);
        let resolved_result = self.resolve_call_result(op.result.as_ref(), result_ty);
        let result_ir_type = resolved_result.ir_type;
        let descriptor = {
            let cc_ref: &crate::call::CallControl = self.callcontrol.as_deref().unwrap();
            cc_ref.getcalldescr(
                op,
                non_void_args,
                result_ir_type,
                OopSpecIndex::None,
                None,
                &mut self.analysis_cache,
                None,
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
        let (args_i, args_r, args_f) = self.rewrite_call_three_lists(func_args, graph_name);
        assert!(
            args_f.is_empty(),
            "force_ir: no float args in conditional_call"
        );
        let result_kind = resolved_result.kind;
        self.stamp_value_kind_from_value_type(graph, op.result.clone(), result_ty);
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
                value: condition_or_value_var,
                funcptr: target.clone(),
                descriptor: descriptor.clone(),
                args_i,
                args_r,
                args_f,
                result_kind,
            }
        } else {
            OpKind::ConditionalCall {
                condition: condition_or_value_var,
                funcptr: target.clone(),
                descriptor: descriptor.clone(),
                args_i,
                args_r,
                args_f,
            }
        };
        // jtransform.py:1681-1682: -live- if calldescr_canraise
        let mut ops = vec![SpaceOperation {
            result: op.result.clone(),
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
        graph: &mut FunctionGraph,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        self._rewrite_op_cond_call(graph, op, target, args, result_ty, graph_name, false)
    }

    /// RPython: `Transformer.rewrite_op_jit_conditional_call_value(op)`
    /// (jtransform.py:1687-1688). Dispatch wrapper kept for structural
    /// parity; pyre's `rewrite_operation` match does not reach it.
    #[allow(dead_code)]
    fn rewrite_op_jit_conditional_call_value(
        &mut self,
        graph: &mut FunctionGraph,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        self._rewrite_op_cond_call(graph, op, target, args, result_ty, graph_name, true)
    }

    /// RPython: `Transformer.rewrite_op_jit_marker(op)` (jtransform.py:1658-1663)
    /// — dispatch portion only. Upstream keys on `op.args[0].value`; pyre
    /// already matched the callee identity in `rewrite_op_direct_call` via
    /// `jit_marker_key_from_target`. Returns `None` when marker state is not
    /// yet wired (no `portal_jd_index`, no `CallControl`, or not enough args
    /// to separate greens from reds) so the caller can fall through to the
    /// regular direct-call handling.
    ///
    /// Upstream also honours `jitdriver.active` (jtransform.py:1661-1662):
    /// when `active=False` the marker is dropped (`return []`). The portal
    /// driver's `active` flag is consulted below before any marker lowering;
    /// pyre seeds it `true` at `setup_jitdriver` and exposes
    /// `CallControl::set_jitdriver_active` to toggle it, matching the
    /// upstream shape.
    fn try_handle_jit_marker(
        &mut self,
        key: JitMarkerKey,
        args: &[crate::flowspace::model::Variable],
        graph: &crate::model::FunctionGraph,
    ) -> Option<Vec<SpaceOperation>> {
        let jitdriver_index = self.portal_jd_index?;
        // jtransform.py:1661-1662 `if not jitdriver.active: return []` — a
        // deactivated portal driver drops its markers before dispatch.
        if let Some(cc) = self.callcontrol.as_deref()
            && let Some(jd) = cc.jitdriver_sd_from_jitdriver(jitdriver_index)
            && !jd.active
        {
            return Some(Vec::new());
        }
        match key {
            JitMarkerKey::LoopHeader | JitMarkerKey::CanEnterJit => {
                // `support.py:77-80` runs its `assert methname ==
                // 'jit_merge_point'` over every marker of a reds='auto'
                // driver, so a `can_enter_jit` on such a driver is a hard
                // error: the auto-detected red set is computed from the
                // jit_merge_point's own block and is not valid for a
                // marker sitting in another block.
                if let Some(cc) = self.callcontrol.as_deref()
                    && let Some(jd) = cc.jitdriver_sd_from_jitdriver(jitdriver_index)
                {
                    assert!(
                        !jd.autoreds,
                        "reds='auto' is supported only for jit drivers which \
                         calls only jit_merge_point. Found a call to {}",
                        match key {
                            JitMarkerKey::CanEnterJit => "can_enter_jit",
                            _ => "loop_header",
                        },
                    );
                }
                // jtransform.py:1723 `handle_jit_marker__can_enter_jit =
                // handle_jit_marker__loop_header`.
                Some(self.handle_jit_marker__loop_header(jitdriver_index))
            }
            JitMarkerKey::JitMergePoint => {
                let (num_greens, autoreds) = {
                    let cc = self.callcontrol.as_deref()?;
                    let jd = cc.jitdriver_sd_from_jitdriver(jitdriver_index)?;
                    (jd.greens.len(), jd.autoreds)
                };
                // Skip the receiver: pyre lowers `driver.jit_merge_point(...)`
                // (`front::mir`) as a method call whose
                // `Call.args[0]` is the `PyPyJitDriver` receiver; the user-facing
                // green/red arguments start at index 1. Upstream's equivalent
                // `jit_marker` op has `args[0]=marker_name_const` and
                // `args[1]=driver_const`, so `op.args[2:]` is the user payload.
                // `jit_marker_key_from_target` already consumed the method name
                // before reaching this branch, leaving only the driver receiver
                // to strip.
                let user_args = match args.split_first() {
                    Some((_receiver, rest)) if rest.len() >= num_greens => rest,
                    _ => return None,
                };
                // jtransform.py:1695 `ops = self.promote_greens(...)` —
                // prepends per-green `-live-` + `{kind}_guard_value` pairs.
                let greens_raw = &user_args[..num_greens];
                // support.py:70-71 guards the portal scan on `autoreds`.
                // Explicit-red drivers retain the existing call-site payload
                // path byte-for-byte.
                let detected_reds =
                    autoreds.then(|| autodetect_jit_markers_redvars(graph, greens_raw));
                let reds_raw = detected_reds.as_deref().unwrap_or(&user_args[num_greens..]);
                if autoreds {
                    let jd = self
                        .callcontrol
                        .as_deref_mut()?
                        .jitdriver_sd_from_jitdriver_mut(jitdriver_index)?;
                    if let Some(numreds) = jd.numreds {
                        assert_eq!(
                            numreds,
                            reds_raw.len(),
                            "there are multiple jit_merge_points with the same jitdriver"
                        );
                    } else {
                        jd.numreds = Some(reds_raw.len());
                    }
                }
                // jtransform.py:1699-1701 `assert isinstance(v,
                // Variable), "Constant specified red in
                // jit_merge_point()"` — a Constant passed as red is
                // rejected.  front::mir materialises every Const
                // operand into a fresh Variable (`emit_constant`), so
                // the Constant-vs-Variable operand distinction is
                // recovered by provenance: a red whose defining op is
                // `Const*` (or the `__str_const` synthetic call) is
                // the image of a source-level Constant red.  Const
                // defining ops pass through jtransform unrewritten
                // (`rewrite_operation` keeps them), so the scan is
                // sound on the in-progress graph.  The graph-wide
                // def-site scan equals the upstream call-operand
                // boundary because graphs are single-assignment
                // (`checkgraph`'s "duplicate variable" assert,
                // flowspace/model.rs): a red holding a constant has
                // its Const def as its ONLY def, and a red that merges
                // a constant with other flow arrives as a fresh block
                // inputarg whose id matches no Const def — exactly the
                // cases upstream sees as Constant resp. Variable
                // operands.  A constant value laundered through a
                // value-copying op is not chased; upstream's
                // `isinstance(v, Variable)` does not chase it either.
                for red in reds_raw {
                    assert!(
                        !is_source_constant_variable(graph, red),
                        "Constant specified red in jit_merge_point()"
                    );
                }
                let mut ops = self.promote_greens(greens_raw);
                let (greens_i, greens_r, greens_f) = split_args_by_kind(greens_raw);
                let (reds_i, reds_r, reds_f) = split_args_by_kind(reds_raw);
                // jtransform.py:1712 final shape is `ops + [op3, op1, op2]`.
                ops.extend(self.handle_jit_marker__jit_merge_point(
                    greens_i, greens_r, greens_f, reds_i, reds_r, reds_f,
                ));
                Some(ops)
            }
        }
    }

    /// RPython: `Transformer.handle_jit_marker__jit_merge_point(op, jitdriver)`
    /// (jtransform.py:1690-1712). Called from `rewrite_op_jit_marker` when the
    /// marker key is `'jit_merge_point'`.
    ///
    /// Upstream takes a `SpaceOperation('jit_marker', [key, jitdriver, *args])`
    /// and `make_three_lists` both green and red args inside. pyre's
    /// `rewrite_op_direct_call` already splits call args by kind, so this port
    /// accepts already-split vectors — the caller feeds `args_i/args_r/args_f`
    /// partitioned at the green/red boundary.
    ///
    /// Returns `[live_preamble, jit_merge_point, live_recursive]`, matching
    /// upstream's `ops + [op3, op1, op2]` shape. The leading `promote_greens`
    /// prefix (`ops`) is empty because `promote_greens` is not yet ported;
    /// greens arrive as Variables/Constants and are forwarded to
    /// the marker unchanged.
    fn handle_jit_marker__jit_merge_point(
        &mut self,
        greens_i: Vec<crate::flowspace::model::Variable>,
        greens_r: Vec<crate::flowspace::model::Variable>,
        greens_f: Vec<crate::flowspace::model::Variable>,
        reds_i: Vec<crate::flowspace::model::Variable>,
        reds_r: Vec<crate::flowspace::model::Variable>,
        reds_f: Vec<crate::flowspace::model::Variable>,
    ) -> Vec<SpaceOperation> {
        // jtransform.py:1691-1692 `assert self.portal_jd is not None`
        let jitdriver_index = self
            .portal_jd_index
            .expect("'jit_merge_point' in non-portal graph!");
        // jtransform.py:1698-1703 — for each red kind-list: every operand
        // must be a `Variable` and no `Variable` may repeat.  The
        // `isinstance(v, Variable)` guard (py:1700) is enforced by
        // provenance in `try_handle_jit_marker` (a Const-defined red is
        // the front-end image of a Constant operand); here only the
        // duplicate-red guard (py:1702 `len(dict.fromkeys(redlist)) ==
        // len(list(redlist))`) remains — a repeated red would alias two
        // live frame slots onto one resume location at the merge point.
        // (The py:1693 `jitdriver is self.portal_jd.jitdriver` mix-up
        // assert is omitted: upstream reads the marker's own driver
        // instance from `op.args[1]`, but pyre's marker is a
        // `CallTarget::Method` on a receiver Variable that carries only
        // the `PyPyJitDriver` *type* — no per-marker driver identity
        // exists to compare against `portal_jd_index`, which is itself
        // derived from the enclosing graph (codewriter.rs
        // `jitdriver_sd_from_portal_graph`).  Portable only once
        // front::mir threads the driver static's identity onto the
        // marker call.)
        for redlist in [&reds_i, &reds_r, &reds_f] {
            let mut seen = std::collections::HashSet::with_capacity(redlist.len());
            for v in redlist {
                assert!(
                    seen.insert(v.id()),
                    "duplicate red variable on jit_merge_point()"
                );
            }
        }
        let merge = SpaceOperation {
            result: None,
            kind: OpKind::JitMergePoint {
                jitdriver_index,
                greens_i,
                greens_r,
                greens_f,
                reds_i,
                reds_r,
                reds_f,
            },
        };
        // jtransform.py:1708-1712 — `op2` live for `do_recursive_call()`,
        // `op3` live for inlined short preambles. Final shape is
        // `ops + [op3, op1, op2]`.
        let live_preamble = SpaceOperation {
            result: None,
            kind: OpKind::Live,
        };
        let live_recursive = SpaceOperation {
            result: None,
            kind: OpKind::Live,
        };
        vec![live_preamble, merge, live_recursive]
    }

    /// RPython: `Transformer.handle_jit_marker__loop_header(op, jitdriver)`
    /// (jtransform.py:1714-1718). `handle_jit_marker__can_enter_jit` aliases
    /// to the same function (jtransform.py:1723); pyre keeps the alias at the
    /// `try_handle_jit_marker` dispatch layer rather than inside this method.
    fn handle_jit_marker__loop_header(&mut self, jitdriver_index: usize) -> Vec<SpaceOperation> {
        vec![SpaceOperation {
            result: None,
            kind: OpKind::LoopHeader { jitdriver_index },
        }]
    }

    /// RPython: `Transformer.rewrite_op_jit_record_known_result(op)`
    /// (jtransform.py:292-313).
    #[allow(dead_code)]
    fn rewrite_op_jit_record_known_result(
        &mut self,
        _graph: &FunctionGraph,
        op: &SpaceOperation,
        target: &CallTarget,
        args: &[crate::flowspace::model::Variable],
        _result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        // jtransform.py:293-295: validate no floats
        for arg in args {
            if self.get_value_kind_var(arg) == 'f' {
                panic!("record_known_result does not support floats");
            }
        }
        // jtransform.py:298-300: calldescr from function (args[1:] → args[0])
        // args[0] = known result, args[1..] = function args
        let result_value = args[0].clone();
        let func_args: &[crate::flowspace::model::Variable] =
            if args.len() > 1 { &args[1..] } else { &[] };
        let result_kind = self.get_value_kind_var(&result_value);
        let result_ir_type = match result_kind {
            'i' => majit_ir::value::Type::Int,
            'r' => majit_ir::value::Type::Ref,
            _ => {
                panic!("record_known_result: unsupported result kind '{result_kind}'");
            }
        };
        let non_void_args = resolve_non_void_arg_types_from_vars(func_args);
        let descriptor = {
            let cc_ref: &crate::call::CallControl = self.callcontrol.as_deref().unwrap();
            cc_ref.getcalldescr(
                op,
                non_void_args,
                result_ir_type,
                OopSpecIndex::None,
                None,
                &mut self.analysis_cache,
                None,
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
        let (args_i, args_r, args_f) = self.rewrite_call_three_lists(func_args, graph_name);
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
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    fn handle_residual_call(
        &mut self,
        graph: &mut FunctionGraph,
        op: &SpaceOperation,
        target: &CallTarget,
        descriptor: CallDescriptor,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        self.handle_residual_call_with_targets(
            graph, op, target, descriptor, args, result_ty, graph_name, None,
        )
    }

    /// RPython `jtransform.py:456-471` + `jtransform.py:547` sidecar:
    /// the `IndirectCallTargets(lst)` passed via `extraargs` rides along
    /// with the residual_call opcode.  This variant exposes the
    /// `indirect_targets` parameter so `handle_regular_indirect_call`
    /// can attach the candidate jitcode list without having to build the
    /// `OpKind::CallResidual` twice.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    fn handle_residual_call_with_targets(
        &mut self,
        graph: &mut FunctionGraph,
        op: &SpaceOperation,
        target: &CallTarget,
        descriptor: CallDescriptor,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
        indirect_targets: Option<crate::model::IndirectCallTargets>,
    ) -> RewriteResult {
        let note_detail = match &indirect_targets {
            Some(t) => format!(
                "call {target} → residual indirect ({} candidates)",
                t.lst.len()
            ),
            None => format!("call {target} → residual"),
        };
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: note_detail,
        });
        self.calls_classified += 1;
        // RPython jtransform.py:467: rewrite_call(op, 'residual_call', ...)
        let (args_i, args_r, args_f) = self.rewrite_call_three_lists(args, graph_name);
        // RPython reads `op.result.concretetype` directly because rtyper
        // has typed every Variable. Pyre's front-end can leave a callee's
        // declared return as `ValueType::Unknown` (re-export shadowing,
        // unresolved cross-crate path); the rtyper's backward-inference
        // pass then assigns a definitive kind via the consumer-op
        // constraint. Honour that kind so the residual_call's
        // `result_kind` matches what every downstream consumer sees,
        // instead of falling back to `'r'` from the Unknown default.
        let result_kind = self.resolve_call_result(op.result.as_ref(), result_ty).kind;
        self.stamp_value_kind_from_value_type(graph, op.result.clone(), result_ty);
        let (funcptr, funcptr_op) = self.direct_funcptr_value(graph, target);
        // RPython jtransform.py:469-470: residual_call followed by -live-
        // if the call can raise or may call jitcodes.
        // jtransform.py:547: `handle_regular_indirect_call` passes
        // `may_call_jitcodes=True`, which forces a trailing `-live-`.
        let can_raise = descriptor.extra_info.check_can_raise(false) || indirect_targets.is_some();
        let mut ops = vec![
            funcptr_op,
            SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::CallResidual {
                    funcptr: CallFuncPtr::Value(funcptr),
                    descriptor,
                    args_i,
                    args_r,
                    args_f,
                    result_kind,
                    indirect_targets,
                },
            },
        ];
        if can_raise {
            ops.push(SpaceOperation {
                result: None,
                kind: OpKind::Live,
            });
        }
        RewriteResult::Replace(ops)
    }

    /// RPython orthodox dispatch for `OpKind::IndirectCall` — line-by-line
    /// port of `jtransform.py:410-412 rewrite_op_indirect_call` +
    /// `jtransform.py:538-553 handle_regular_indirect_call`.
    ///
    /// `funcptr` is the runtime Variable already produced by the
    /// rtyper-equivalent layer (`translator/rtyper/rpbc.rs::lower_indirect_calls`),
    /// so this method does NOT synthesize anything — it emits exactly:
    ///
    /// ```text
    /// [Live, IntGuardValue(funcptr, 'i'),
    ///  CallResidual { funcptr: Value(funcptr),
    ///                 indirect_targets: Some(IndirectCallTargets { lst }), .. },
    ///  Live]    // trailing -live- because may_call_jitcodes=true
    /// ```
    ///
    /// `lst` is built from `candidates` via `cc.get_jitcode(p)` which
    /// returns the `Arc<JitCode>` shell from `CallControl::jitcodes`.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    fn lower_indirect_call_op(
        &mut self,
        op: &SpaceOperation,
        funcptr: &crate::flowspace::model::Variable,
        args: &[crate::flowspace::model::Variable],
        graphs: Option<&[crate::parse::CallPath]>,
        result_ty: &ValueType,
        graph_name: &str,
        graph: &mut crate::model::FunctionGraph,
    ) -> RewriteResult {
        let materialized_funcptr = self.materialize_fn_const_funcptr_value(graph, funcptr);
        let funcptr_value = materialized_funcptr
            .as_ref()
            .map(|(var, _)| var)
            .unwrap_or(funcptr)
            .clone();
        let (args_i, args_r, args_f) = self.rewrite_call_three_lists(args, graph_name);
        let resolved_result = self.resolve_call_result(op.result.as_ref(), result_ty);
        let result_kind = resolved_result.kind;
        self.stamp_value_kind_from_value_type(graph, op.result.clone(), result_ty);
        let non_void_args = resolve_non_void_arg_types_from_vars(args);
        let result_ir_type = resolved_result.ir_type;
        let cc_mut = self
            .callcontrol
            .as_mut()
            .expect("rewrite_op_indirect_call requires &mut CallControl");
        let descriptor = cc_mut.getcalldescr(
            op,
            non_void_args,
            result_ir_type,
            OopSpecIndex::None,
            None,
            &mut self.analysis_cache,
            None,
        );
        match cc_mut.guess_call_kind(op) {
            crate::call::CallKind::Regular => {
                let candidates = cc_mut
                    .graphs_from(op)
                    .expect("regular indirect call must have candidate graphs");
                let lst: Vec<crate::jitcode::JitCodeHandle> = candidates
                    .iter()
                    .map(|p| crate::jitcode::JitCodeHandle::new(cc_mut.get_jitcode(p)))
                    .collect();

                // jtransform.py:545-552 emit sequence:
                //   op0 = SpaceOperation('-live-', [], None)
                //   op1 = SpaceOperation('int_guard_value', [op.args[0]], None)
                //   op2 = self.handle_residual_call(op, [IndirectCallTargets(lst)], True)
                // then [op0, op1] + op2 (op2 is itself [residual_call, '-live-'])
                let mut ops = Vec::<SpaceOperation>::with_capacity(5);
                if let Some((_, funcptr_op)) = materialized_funcptr.clone() {
                    ops.push(funcptr_op);
                }
                ops.push(SpaceOperation {
                    result: None,
                    kind: OpKind::Live,
                });
                ops.push(SpaceOperation {
                    result: None,
                    kind: OpKind::GuardValue {
                        value: funcptr_value.clone(),
                        kind_char: 'i',
                    },
                });
                ops.push(SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::CallResidual {
                        funcptr: CallFuncPtr::Value(funcptr_value.clone()),
                        descriptor,
                        args_i,
                        args_r,
                        args_f,
                        result_kind,
                        indirect_targets: Some(crate::model::IndirectCallTargets { lst }),
                    },
                });
                // jtransform.py:469-470: residual_call followed by -live-
                // because `may_call_jitcodes=True` for the regular-indirect path.
                ops.push(SpaceOperation {
                    result: None,
                    kind: OpKind::Live,
                });

                self.calls_classified += 1;
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: format!("indirect call → {} candidates", candidates.len()),
                });
                RewriteResult::Replace(ops)
            }
            crate::call::CallKind::Residual => {
                let can_raise = descriptor.extra_info.check_can_raise(false);
                self.calls_classified += 1;
                self.notes.push(GraphTransformNote {
                    function: graph_name.to_string(),
                    detail: match graphs {
                        Some(graphs) => {
                            format!("call <indirect> → residual family ({} impls)", graphs.len())
                        }
                        None => "call <indirect> → residual unknown family".to_string(),
                    },
                });
                let mut ops = Vec::new();
                if let Some((_, funcptr_op)) = materialized_funcptr {
                    ops.push(funcptr_op);
                }
                ops.push(SpaceOperation {
                    result: op.result.clone(),
                    kind: OpKind::CallResidual {
                        funcptr: CallFuncPtr::Value(funcptr_value),
                        descriptor,
                        args_i,
                        args_r,
                        args_f,
                        result_kind,
                        indirect_targets: None,
                    },
                });
                if can_raise {
                    ops.push(SpaceOperation {
                        result: None,
                        kind: OpKind::Live,
                    });
                }
                RewriteResult::Replace(ops)
            }
            crate::call::CallKind::Builtin | crate::call::CallKind::Recursive => {
                unreachable!("indirect calls cannot classify as builtin/recursive")
            }
        }
    }

    /// RPython: elidable call — pure function, result depends only on args.
    /// RPython jtransform.py:546-562.
    ///
    /// `target` is the funcptr identity per jtransform.py:457.
    #[allow(dead_code)]
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    fn handle_elidable_call(
        &mut self,
        graph: &mut FunctionGraph,
        op: &SpaceOperation,
        target: &CallTarget,
        descriptor: CallDescriptor,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("call {target} → elidable"),
        });
        self.calls_classified += 1;
        let (args_i, args_r, args_f) = self.rewrite_call_three_lists(args, graph_name);
        let result_kind = self.resolve_call_result(op.result.as_ref(), result_ty).kind;
        self.stamp_value_kind_from_value_type(graph, op.result.clone(), result_ty);
        let (funcptr, funcptr_op) = self.direct_funcptr_value(graph, target);
        RewriteResult::Replace(vec![
            funcptr_op,
            SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::CallElidable {
                    funcptr: CallFuncPtr::Value(funcptr),
                    descriptor,
                    args_i,
                    args_r,
                    args_f,
                    result_kind,
                },
            },
        ])
    }

    /// RPython: may-force call — can trigger GC or force virtualizables.
    /// RPython jtransform.py:609-625.
    ///
    /// `target` is the funcptr identity per jtransform.py:457.
    #[allow(dead_code)]
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython translation routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and ownership"
    )]
    fn handle_may_force_call(
        &mut self,
        graph: &mut FunctionGraph,
        op: &SpaceOperation,
        target: &CallTarget,
        descriptor: CallDescriptor,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
        graph_name: &str,
    ) -> RewriteResult {
        self.notes.push(GraphTransformNote {
            function: graph_name.to_string(),
            detail: format!("call {target} → may_force"),
        });
        self.calls_classified += 1;
        let (args_i, args_r, args_f) = self.rewrite_call_three_lists(args, graph_name);
        let result_kind = self.resolve_call_result(op.result.as_ref(), result_ty).kind;
        self.stamp_value_kind_from_value_type(graph, op.result.clone(), result_ty);
        let (funcptr, funcptr_op) = self.direct_funcptr_value(graph, target);
        // RPython: call_may_force always followed by -live-
        RewriteResult::Replace(vec![
            funcptr_op,
            SpaceOperation {
                result: op.result.clone(),
                kind: OpKind::CallMayForce {
                    funcptr: CallFuncPtr::Value(funcptr),
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

    /// Decide whether a `direct_call` is a transparent Rust prelude
    /// constructor that the frontend has already proved is not a real
    /// callable. Returns `true` iff every requirement holds, so the caller can
    /// safely emit `RewriteResult::Identity(args[0])`.
    ///
    /// Requirements:
    /// 1. Target is `CallTarget::SyntheticTransparentCtor`.
    /// 2. `args.len() == 1`.
    /// 3. The arg's resolved IR kind equals the result's IR kind. A
    ///    transparent wrapper preserves representation (`r → r`,
    ///    `i → i`, …); a kind mismatch (e.g. `i → r`) means the
    ///    Rust call is doing real boxing and must not be elided.
    fn is_synthetic_result_option_ctor(
        &self,
        target: &CallTarget,
        args: &[crate::flowspace::model::Variable],
        result_ty: &ValueType,
    ) -> bool {
        if args.len() != 1 {
            return false;
        }
        let CallTarget::SyntheticTransparentCtor {
            name,
            owner_path: _,
        } = target
        else {
            return false;
        };
        // Discriminator is the leaf name: `Ok`/`Err`/`Some` are the
        // single-arg transparent wrappers PyPy treats as identity
        // value-level (constructed by `rpython/rtyper/lltypesystem/
        // rtagged.py` / `rtyper/llinterp.py` PBC paths, never
        // materialised at runtime).  The producer
        // (`front::mir`) accepts both bare (`Ok`) and qualified
        // (`Result::Ok`, `std::option::Option::Some`) spellings —
        // qualified forms carry their leading segments as
        // `owner_path`; both must elide
        // identically because PyPy doesn't distinguish call-site
        // spelling at the SSA layer.  Unit variants like
        // `StepResult::Continue` route through the same
        // `SyntheticTransparentCtor` variant but have a different
        // `name` and therefore skip this arm.
        if !matches!(name.as_str(), "Ok" | "Err" | "Some") {
            return false;
        }
        // arg/result IR-kind parity. `resolve_non_void_arg_types_from_vars`
        // returns `Type::Ref` when type_state is missing or the
        // value is unknown — that's the same default
        // `value_type_to_kind` applies for an `Unknown` result, so
        // the comparison stays sound under partial type info.
        let arg_types = resolve_non_void_arg_types_from_vars(args);
        let arg_ir = arg_types
            .first()
            .copied()
            .unwrap_or(majit_ir::value::Type::Ref);
        let result_ir = value_type_to_ir_type(result_ty);
        arg_ir == result_ir
    }
}

/// `jtransform.py:196-234 Transformer.optimize_goto_if_not` — fuse a
/// comparison op into the block's `exitswitch`.
///
/// Replaces `v = int_gt(x, y); exitswitch = v` with the fused
/// `exitswitch = ('int_gt', x, y, '-live-before')`.  The `-live-before`
/// marker is implicit in pyre's [`ExitSwitch::Fused`] (re-applied at
/// emit time, `flatten.py:248-253`) so it is not stored in `args`.
///
/// Faithful port, line-for-line:
/// ```python
/// def optimize_goto_if_not(self, block):
///     if len(block.exits) != 2:
///         return False
///     v = block.exitswitch
///     if (block.canraise or isinstance(v, tuple)
///             or v.concretetype != lltype.Bool):
///         return False
///     for op in block.operations[::-1]:
///         for arg in op.args:
///             if arg == v:
///                 return False
///             if isinstance(arg, ListOfKind) and v in arg.content:
///                 return False
///         if v is op.result:
///             if op.opname not in (...comparison opnames...):
///                 return False
///             block.operations.remove(op)
///             block.exitswitch = (op.opname,) + tuple(op.args)
///             block.exitswitch += ('-live-before',)
///             for link in block.exits:
///                 while v in link.args:
///                     index = link.args.index(v)
///                     link.args[index] = Constant(link.llexitcase, lltype.Bool)
///             return True
///     return False
/// ```
///
/// Notes on faithful adaptations to pyre's IR:
/// - `isinstance(v, tuple)` (an already-fused switch) maps to "the
///   `exitswitch` is not a plain [`ExitSwitch::Value`]" — i.e.
///   `Fused` / `LastException` / `None` all return `false`, exactly as
///   upstream skips a tuple / can-raise / unset switch.
/// - `v.concretetype != lltype.Bool` reads the backing Variable's raw
///   `LowLevelType` (`Variable::concretetype()`), checking
///   [`LowLevelType::Bool`] directly.  Going through
///   [`FunctionGraph::concretetype_of`] / [`ConcreteType`] would be
///   wrong here because that projection collapses `Bool` into
///   `ConcreteType::Signed` (`getkind`, `history.py:45-71`); the Bool
///   distinction only survives on the raw lltype cell.
/// - `for arg in op.args` (incl. the `ListOfKind` content scan) maps to
///   [`crate::inline::op_variable_refs`], which already flattens every
///   operand Variable of any op — including aggregate operands
///   (`NewTuple`, `JitMergePoint`, ...) that are pyre's `ListOfKind`
///   analogue — so the explicit `ListOfKind`/`v in arg.content` arm is
///   subsumed.
/// - `Constant(link.llexitcase, lltype.Bool)`: pyre's `link.llexitcase`
///   may be `None` pre-rtyper, but the bool branch value is reliably in
///   `link.exitcase` as [`ExitCase::Bool`] (the same source
///   `with_llexitcase_from_exitcase` reads, `model.rs:1244-1252`); the
///   substituted constant is
///   `Constant::with_concretetype(ConstValue::Bool(b), lltype.Bool)`,
///   carrying the `lltype.Bool` concretetype upstream stamps.
//
// the GotoIfNotOp lowering: called from `optimize_block` (jtransform.py:123); the fused
// `ExitSwitch::Fused` is lowered to `FlatOp::GotoIfNotOp` in flatten.
fn optimize_goto_if_not(graph: &mut FunctionGraph, block_idx: usize) -> bool {
    use crate::flowspace::model::{ConstValue, Constant};
    use crate::model::{ExitCase, ExitSwitch};
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;

    // `if len(block.exits) != 2: return False`
    if graph.blocks[block_idx].exits.len() != 2 {
        return false;
    }
    // `v = block.exitswitch`
    //
    // `isinstance(v, tuple)` / can-raise (`LastException`) / unset
    // switch all collapse to "not a plain `Value(v)`".
    let v = match &graph.blocks[block_idx].exitswitch {
        Some(ExitSwitch::Value(var)) => var.clone(),
        _ => return false,
    };
    // `if block.canraise or ...: return False`
    if graph.blocks[block_idx].canraise() {
        return false;
    }
    // `or v.concretetype != lltype.Bool: return False`
    if v.concretetype() != Some(LowLevelType::Bool) {
        return false;
    }

    // `for op in block.operations[::-1]:`
    let ops_len = graph.blocks[block_idx].operations.len();
    for op_idx in (0..ops_len).rev() {
        // `for arg in op.args: if arg == v (or v in ListOfKind): return False`
        let reads =
            crate::inline::op_variable_refs(&graph.blocks[block_idx].operations[op_idx].kind);
        if reads.contains(&v) {
            return false;
        }
        // `if v is op.result:`
        if graph.blocks[block_idx].operations[op_idx].result.as_ref() == Some(&v) {
            // `if op.opname not in (...): return False` else fuse.
            let Some((opname, args)) =
                goto_if_not_fusable(&graph.blocks[block_idx].operations[op_idx].kind)
            else {
                return false;
            };
            // `block.operations.remove(op)`
            graph.blocks[block_idx].operations.remove(op_idx);
            // `block.exitswitch = (op.opname,) + tuple(op.args) + ('-live-before',)`
            // (the `-live-before` marker is implicit in `Fused`).
            graph.blocks[block_idx].exitswitch = Some(ExitSwitch::Fused { opname, args });
            // `for link in block.exits:
            //      while v in link.args:
            //          link.args[index] = Constant(link.llexitcase, lltype.Bool)`
            //
            // The substitution only matters when the fused result `v`
            // flows into a successor block (rare); links that do not
            // carry `v` are left untouched.  The replacement bool is
            // read `llexitcase`-first then `exitcase` — the same source
            // order as `flatten.rs::bool_llexitcase` — so post-rtyper
            // graphs (bool in `llexitcase`) and pre-rtyper semantic
            // graphs (bool in `exitcase`) both resolve.
            for link in graph.blocks[block_idx].exits.iter_mut() {
                if !link.args.iter().any(|arg| arg.as_variable() == Some(&v)) {
                    continue;
                }
                let bool_const = match &link.llexitcase {
                    Some(ConstValue::Bool(b)) => *b,
                    _ => match &link.exitcase {
                        Some(ExitCase::Bool(b)) => *b,
                        Some(ExitCase::Const(ConstValue::Bool(b))) => *b,
                        other => panic!(
                            "optimize_goto_if_not: fused bool branch link carrying the \
                             result variable lacks a bool ll/exitcase \
                             (jtransform.py:230 link.llexitcase), got {other:?}"
                        ),
                    },
                };
                for arg in link.args.iter_mut() {
                    if arg.as_variable() == Some(&v) {
                        // `Constant(link.llexitcase, lltype.Bool)` —
                        // bool value carries the `lltype.Bool` concretetype.
                        *arg = LinkArg::Const(Constant::with_concretetype(
                            ConstValue::Bool(bool_const),
                            LowLevelType::Bool,
                        ));
                    }
                }
            }
            // `return True`
            return true;
        }
    }
    // `return False`
    false
}

/// `jtransform.py:206-209` supported-opname gate for
/// [`optimize_goto_if_not`].  Returns the RPython opname and the op's
/// operand Variables (`tuple(op.args)`) when the op is one of the
/// comparison / boolean-test opcodes that may be fused into a guard;
/// `None` for any other op (upstream's `return False`).
///
/// The Bool-producing compares are pyre [`OpKind::BinOp`]s; the unary
/// `int_is_zero` / `int_is_true` / `ptr_iszero` / `ptr_nonzero` tests
/// are [`OpKind::UnaryOp`]s.
fn goto_if_not_fusable(kind: &OpKind) -> Option<(String, Vec<crate::flowspace::model::Variable>)> {
    match kind {
        OpKind::BinOp { op, lhs, rhs, .. }
            if matches!(
                op.as_str(),
                "int_lt"
                    | "int_le"
                    | "int_eq"
                    | "int_ne"
                    | "int_gt"
                    | "int_ge"
                    | "float_lt"
                    | "float_le"
                    | "float_eq"
                    | "float_ne"
                    | "float_gt"
                    | "float_ge"
                    | "ptr_eq"
                    | "ptr_ne"
            ) =>
        {
            Some((op.clone(), vec![lhs.clone(), rhs.clone()]))
        }
        OpKind::UnaryOp { op, operand, .. }
            if matches!(
                op.as_str(),
                "int_is_zero" | "int_is_true" | "ptr_iszero" | "ptr_nonzero"
            ) =>
        {
            Some((op.clone(), vec![operand.clone()]))
        }
        _ => None,
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
fn resolve_non_void_arg_types_from_vars(
    args: &[crate::flowspace::model::Variable],
) -> Vec<majit_ir::value::Type> {
    args.iter()
        .filter_map(|var| {
            let kind = match crate::model::FunctionGraph::concretetype_of(var) {
                crate::codewriter::type_state::ConcreteType::Signed => 'i',
                crate::codewriter::type_state::ConcreteType::GcRef => 'r',
                crate::codewriter::type_state::ConcreteType::Float => 'f',
                crate::codewriter::type_state::ConcreteType::Void => 'v',
                crate::codewriter::type_state::ConcreteType::Unknown => 'r',
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
        // RPython `getkind(BOOL_TYPE)` / `getkind(Unsigned)` both
        // return `'int'` (ll Bool / Unsigned share register class
        // with Signed at the codewriter register-kind layer).
        ValueType::Int | ValueType::Unsigned | ValueType::Bool | ValueType::State => 'i',
        ValueType::Ref(_) | ValueType::Str | ValueType::Unknown => 'r',
        ValueType::Float => 'f',
        ValueType::Void => 'v',
        ValueType::Int128 | ValueType::UInt128 => {
            panic!("getkind: 128-bit integer type is too large (history.py:62)")
        }
    }
}

/// RPython `rfloat._rtype_template` calls `hop.inputargs(Float, Float)`,
/// which only coerces `Signed → Float`.  Pyre's float arms enter
/// when both operands fall into this domain.
fn is_float_rewrite_domain(kind: char) -> bool {
    matches!(kind, 'i' | 'f')
}

/// Rust assignment operators have no separate RPython flow op.  Include them
/// here so float-domain rewrites run before the generic assignment collapse.
fn canonical_float_arith_binop(op: &str) -> Option<&'static str> {
    match op {
        "add" | "add_assign" => Some("add"),
        "sub" | "sub_assign" => Some("sub"),
        "mul" | "mul_assign" => Some("mul"),
        // `front::mir::canonical_binop_label` collapses Rust `/`
        // (MIR `Div`) to "floordiv"; over floats that operator is
        // true division, so both labels land on `float_truediv`.
        // The front end re-labels a float-result `/` to "truediv"
        // (mirroring RPython's flowspace opname) before the prepass
        // annotator, so this arm also accepts the already-canonical
        // "truediv" / "inplace_truediv" spellings.
        // Integer "floordiv" never reaches this arm — the guard
        // requires a float operand or a Float result.
        "div" | "div_assign" | "floordiv" | "truediv" | "inplace_truediv" => Some("div"),
        _ => None,
    }
}

fn canonical_float_mod_binop(op: &str) -> Option<&'static str> {
    match op {
        "mod" | "mod_assign" => Some("mod"),
        _ => None,
    }
}

fn canonical_assign_binop(op: &str) -> Option<&'static str> {
    match op {
        "add_assign" => Some("add"),
        "sub_assign" => Some("sub"),
        "mul_assign" => Some("mul"),
        "div_assign" => Some("div"),
        "mod_assign" => Some("mod"),
        "bitand_assign" => Some("and"),
        "bitor_assign" => Some("or"),
        "bitxor_assign" => Some("xor"),
        "rshift_assign" => Some("rshift"),
        "lshift_assign" => Some("lshift"),
        _ => None,
    }
}

/// Convert codewriter ValueType to IR Type.
///
/// RPython: `x.concretetype` → lltype mapping.
/// Used by getcalldescr to build NON_VOID_ARGS and RESULT types.
fn value_type_to_ir_type(ty: &ValueType) -> majit_ir::value::Type {
    match ty {
        ValueType::Int | ValueType::Unsigned | ValueType::Bool | ValueType::State => {
            majit_ir::value::Type::Int
        }
        ValueType::Ref(_) | ValueType::Str | ValueType::Unknown => majit_ir::value::Type::Ref,
        ValueType::Float => majit_ir::value::Type::Float,
        ValueType::Void => majit_ir::value::Type::Void,
        ValueType::Int128 | ValueType::UInt128 => {
            panic!("getkind: 128-bit integer type is too large (history.py:62)")
        }
    }
}

fn fn_const_target_for_var(
    graph: &FunctionGraph,
    var: &crate::flowspace::model::Variable,
    depth: usize,
) -> Option<CallTarget> {
    if depth > 8 {
        return None;
    }
    let (block_id, index) = crate::front::mir::resolve_to_producer_op(graph, var)?;
    let block = graph.blocks.iter().find(|block| block.id == block_id)?;
    let op = block.operations.get(index)?;
    fn_const_target_from_op(graph, op, depth + 1)
}

fn fn_const_target_from_op(
    graph: &FunctionGraph,
    op: &SpaceOperation,
    depth: usize,
) -> Option<CallTarget> {
    match &op.kind {
        OpKind::Call { target, args, .. }
            if args.is_empty() && crate::model::fn_const_segments(target).is_some() =>
        {
            Some(target.clone())
        }
        OpKind::UnaryOp { op, operand, .. } if op == "same_as" => {
            fn_const_target_for_var(graph, operand, depth)
        }
        OpKind::Hint { value, .. } => fn_const_target_for_var(graph, value, depth),
        OpKind::FieldRead { base, field, .. } => {
            let index = tuple_pos_field_index(&field.name)?;
            if let Some(target) = fn_const_target_from_tuple_arg(graph, base, index, depth) {
                return Some(target);
            }
            fn_const_target_from_field_write(graph, base, field, depth)
        }
        _ => None,
    }
}

fn fn_const_target_from_tuple_arg(
    graph: &FunctionGraph,
    base: &crate::flowspace::model::Variable,
    index: usize,
    depth: usize,
) -> Option<CallTarget> {
    let (block_id, op_index) = crate::front::mir::resolve_to_producer_op(graph, base)?;
    let block = graph.blocks.iter().find(|block| block.id == block_id)?;
    let op = block.operations.get(op_index)?;
    let OpKind::NewTuple { args } = &op.kind else {
        return None;
    };
    let arg = args.get(index)?;
    fn_const_target_for_var(graph, arg, depth)
}

fn fn_const_target_from_field_write(
    graph: &FunctionGraph,
    base: &crate::flowspace::model::Variable,
    field: &crate::model::FieldDescriptor,
    depth: usize,
) -> Option<CallTarget> {
    // The scan is flow-insensitive, so two arms writing different function
    // pointers to the same field are indistinguishable here.  Taking the
    // first would bake one arm's address into a call the other arm reaches,
    // so an ambiguous initializer declines to materialise a constant and the
    // call stays indirect.
    let mut found: Option<CallTarget> = None;
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::FieldWrite {
                base: write_base,
                field: write_field,
                value,
                ..
            } = &op.kind
            else {
                continue;
            };
            if write_base != base || write_field != field {
                continue;
            }
            let target = value
                .as_variable()
                .and_then(|value| fn_const_target_for_var(graph, value, depth))?;
            match &found {
                Some(seen) if *seen != target => return None,
                Some(_) => {}
                None => found = Some(target),
            }
        }
    }
    found
}

fn tuple_pos_field_index(name: &str) -> Option<usize> {
    name.strip_prefix("__pos_")?.parse().ok()
}

/// Convert a CallTarget to a CallPath for jitcode lookup.
fn target_to_call_path(target: &CallTarget) -> crate::parse::CallPath {
    match target {
        CallTarget::FunctionPath { segments } => {
            let segments = crate::model::fn_const_segments(target).unwrap_or(segments.as_slice());
            crate::parse::CallPath::from_segments(segments.iter().map(String::as_str))
        }
        CallTarget::Method { name, .. } => crate::parse::CallPath::from_segments([name.as_str()]),
        CallTarget::SyntheticTransparentCtor { name, owner_path } => {
            let mut segs: Vec<&str> = owner_path.iter().map(String::as_str).collect();
            segs.push(name.as_str());
            crate::parse::CallPath::from_segments(segs)
        }
        // RPython: an indirect_call has no single jitcode-lookup path —
        // the family is handled via the op-based `graphs_from(op)` +
        // `IndirectCallTargets` sidecar.  This fallback returns a stub
        // path only reached by callers that don't distinguish; the real
        // consumer (`handle_regular_indirect_call`) uses the family path
        // directly.
        CallTarget::Indirect {
            trait_root,
            method_name,
        } => crate::parse::CallPath::from_segments([trait_root.as_str(), method_name.as_str()]),
        CallTarget::UnsupportedExpr => crate::parse::CallPath::from_segments(["<unsupported>"]),
    }
}

/// RPython `jtransform.py:264-275 _renamings_get(self, v)` — follow the
/// alias chain to the canonical Variable.  Variable-keyed: RPython
/// `self._renamings` is a `{Variable: Variable}` dict
/// (`jtransform.py:71`).
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn resolve_alias(
    value: &crate::flowspace::model::Variable,
    aliases: &std::collections::HashMap<
        crate::flowspace::model::Variable,
        crate::flowspace::model::Variable,
    >,
) -> crate::flowspace::model::Variable {
    let mut cur = value.clone();
    while let Some(next) = aliases.get(&cur).cloned() {
        if next == cur {
            break;
        }
        cur = next;
    }
    cur
}

#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn remap_value(
    value: &crate::flowspace::model::Variable,
    aliases: &std::collections::HashMap<
        crate::flowspace::model::Variable,
        crate::flowspace::model::Variable,
    >,
) -> crate::flowspace::model::Variable {
    resolve_alias(value, aliases)
}

#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn remap_call_funcptr(
    funcptr: &CallFuncPtr,
    aliases: &std::collections::HashMap<
        crate::flowspace::model::Variable,
        crate::flowspace::model::Variable,
    >,
) -> CallFuncPtr {
    match funcptr {
        CallFuncPtr::Target(target) => CallFuncPtr::Target(target.clone()),
        CallFuncPtr::Value(var) => CallFuncPtr::Value(remap_value(var, aliases)),
    }
}

#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn remap_op(
    op: &SpaceOperation,
    aliases: &std::collections::HashMap<
        crate::flowspace::model::Variable,
        crate::flowspace::model::Variable,
    >,
) -> SpaceOperation {
    let kind = match &op.kind {
        OpKind::Input { .. }
        | OpKind::ConstInt(_)
        | OpKind::ConstInt128(_)
        | OpKind::ConstUInt128(_)
        | OpKind::ConstBool(_)
        | OpKind::ConstSymbolic { .. }
        | OpKind::ConstFloat(_)
        | OpKind::ConstStr(_)
        | OpKind::ConstRef(_)
        | OpKind::ConstRefNull
        | OpKind::ConstNone
        | OpKind::ConstRefAddr(_)
        | OpKind::CurrentTraceLength
        | OpKind::Live
        | OpKind::LoopHeader { .. }
        | OpKind::Abort { .. }
        | OpKind::LoadStatic { .. }
        | OpKind::New { .. }
        | OpKind::NewWithVtable { .. } => op.kind.clone(),
        OpKind::NewTuple { args } => OpKind::NewTuple {
            args: args.iter().map(|a| remap_value(a, aliases)).collect(),
        },
        OpKind::NewList { args } => OpKind::NewList {
            args: args.iter().map(|a| remap_value(a, aliases)).collect(),
        },
        OpKind::NewArrayClear {
            length,
            item_ty,
            array_type_id,
        } => OpKind::NewArrayClear {
            length: remap_value(length, aliases),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
        },
        OpKind::NewListClear {
            length,
            item_ty,
            array_type_id,
        } => OpKind::NewListClear {
            length: remap_value(length, aliases),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
        },
        OpKind::GetSlice { args } => OpKind::GetSlice {
            args: args.iter().map(|a| remap_value(a, aliases)).collect(),
        },
        OpKind::LoweredBlackholeOp { opname, args } => OpKind::LoweredBlackholeOp {
            opname: opname.clone(),
            args: args.iter().map(|a| remap_value(a, aliases)).collect(),
        },
        OpKind::GuardValue { value, kind_char } => OpKind::GuardValue {
            value: remap_value(value, aliases),
            kind_char: *kind_char,
        },
        OpKind::VtableMethodPtr {
            receiver,
            trait_root,
            method_name,
        } => OpKind::VtableMethodPtr {
            receiver: remap_value(receiver, aliases),
            trait_root: trait_root.clone(),
            method_name: method_name.clone(),
        },
        OpKind::VableForce { base } => OpKind::VableForce {
            base: remap_value(base, aliases),
        },
        OpKind::Hint { value, kind } => OpKind::Hint {
            value: remap_value(value, aliases),
            kind: *kind,
        },
        OpKind::JitMergePoint {
            jitdriver_index,
            greens_i,
            greens_r,
            greens_f,
            reds_i,
            reds_r,
            reds_f,
        } => {
            let remap_var = |var: &crate::flowspace::model::Variable| remap_value(var, aliases);
            OpKind::JitMergePoint {
                jitdriver_index: *jitdriver_index,
                greens_i: greens_i.iter().map(remap_var).collect(),
                greens_r: greens_r.iter().map(remap_var).collect(),
                greens_f: greens_f.iter().map(remap_var).collect(),
                reds_i: reds_i.iter().map(remap_var).collect(),
                reds_r: reds_r.iter().map(remap_var).collect(),
                reds_f: reds_f.iter().map(remap_var).collect(),
            }
        }
        OpKind::IndirectCall {
            funcptr,
            args,
            graphs,
            result_ty,
        } => {
            let remap_var = |var: &crate::flowspace::model::Variable| remap_value(var, aliases);
            OpKind::IndirectCall {
                funcptr: remap_var(funcptr),
                args: args.iter().map(remap_var).collect(),
                graphs: graphs.clone(),
                result_ty: result_ty.clone(),
            }
        }
        OpKind::RecordQuasiImmutField {
            base,
            field,
            mutate_field,
        } => OpKind::RecordQuasiImmutField {
            base: remap_value(base, aliases),
            field: field.clone(),
            mutate_field: mutate_field.clone(),
        },
        OpKind::FieldRead {
            base,
            field,
            ty,
            pure,
        } => OpKind::FieldRead {
            base: remap_value(base, aliases),
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
            base: remap_value(base, aliases),
            field: field.clone(),
            value: value.map_value(|v| remap_value(v, aliases)),
            ty: ty.clone(),
        },
        OpKind::ArrayRead {
            base,
            index,
            item_ty,
            array_type_id,
            nolength,
            pure,
        } => OpKind::ArrayRead {
            base: remap_value(base, aliases),
            index: remap_value(index, aliases),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
            nolength: *nolength,
            // Preserve the source foldable/immutable flag through the
            // alias remap — never hardcode it (rlist.py:724
            // ll_getitem_foldable_nonneg).
            pure: *pure,
        },
        OpKind::ArrayLen {
            base,
            array_type_id,
            nolength,
        } => OpKind::ArrayLen {
            base: remap_value(base, aliases),
            array_type_id: array_type_id.clone(),
            nolength: *nolength,
        },
        OpKind::ArrayWrite {
            base,
            index,
            value,
            item_ty,
            array_type_id,
            nolength,
        } => OpKind::ArrayWrite {
            base: remap_value(base, aliases),
            index: remap_value(index, aliases),
            value: value.map_value(|v| remap_value(v, aliases)),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
            nolength: *nolength,
        },
        OpKind::InteriorFieldRead {
            base,
            index,
            field,
            item_ty,
            array_type_id,
        } => OpKind::InteriorFieldRead {
            base: remap_value(base, aliases),
            index: remap_value(index, aliases),
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
            base: remap_value(base, aliases),
            index: remap_value(index, aliases),
            field: field.clone(),
            value: remap_value(value, aliases),
            item_ty: item_ty.clone(),
            array_type_id: array_type_id.clone(),
        },
        OpKind::Call {
            target,
            args,
            result_ty,
        } => {
            let remap_var = |var: &crate::flowspace::model::Variable| remap_value(var, aliases);
            OpKind::Call {
                target: target.clone(),
                args: args.iter().map(remap_var).collect(),
                result_ty: result_ty.clone(),
            }
        }
        OpKind::GuardTrue { cond } => OpKind::GuardTrue {
            cond: remap_value(cond, aliases),
        },
        OpKind::GuardFalse { cond } => OpKind::GuardFalse {
            cond: remap_value(cond, aliases),
        },
        OpKind::VableFieldRead {
            base,
            field_index,
            ty,
        } => OpKind::VableFieldRead {
            base: remap_value(base, aliases),
            field_index: *field_index,
            ty: ty.clone(),
        },
        OpKind::VableFieldWrite {
            base,
            field_index,
            value,
            ty,
        } => OpKind::VableFieldWrite {
            base: remap_value(base, aliases),
            field_index: *field_index,
            value: value.map_value(|v| remap_value(v, aliases)),
            ty: ty.clone(),
        },
        OpKind::VableArrayRead {
            base,
            array_index,
            elem_index,
            item_ty,
            array_itemsize,
            array_is_signed,
        } => OpKind::VableArrayRead {
            base: remap_value(base, aliases),
            array_index: *array_index,
            elem_index: remap_value(elem_index, aliases),
            item_ty: item_ty.clone(),
            array_itemsize: *array_itemsize,
            array_is_signed: *array_is_signed,
        },
        OpKind::VableArrayWrite {
            base,
            array_index,
            elem_index,
            value,
            item_ty,
            array_itemsize,
            array_is_signed,
        } => OpKind::VableArrayWrite {
            base: remap_value(base, aliases),
            array_index: *array_index,
            elem_index: remap_value(elem_index, aliases),
            value: remap_value(value, aliases),
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
            lhs: remap_value(lhs, aliases),
            rhs: remap_value(rhs, aliases),
            result_ty: result_ty.clone(),
        },
        OpKind::UnaryOp {
            op,
            operand,
            result_ty,
        } => OpKind::UnaryOp {
            op: op.clone(),
            operand: remap_value(operand, aliases),
            result_ty: result_ty.clone(),
        },
        OpKind::JitDebug { args } => OpKind::JitDebug {
            args: args.iter().map(|var| remap_value(var, aliases)).collect(),
        },
        OpKind::RecordKnownResult {
            result_value,
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => {
            let remap_var = |var: &crate::flowspace::model::Variable| remap_value(var, aliases);
            OpKind::RecordKnownResult {
                result_value: remap_var(result_value),
                funcptr: funcptr.clone(),
                descriptor: descriptor.clone(),
                args_i: args_i.iter().map(remap_var).collect(),
                args_r: args_r.iter().map(remap_var).collect(),
                args_f: args_f.iter().map(remap_var).collect(),
                result_kind: *result_kind,
            }
        }
        OpKind::AssertGreen { value, kind_char } => OpKind::AssertGreen {
            value: remap_value(value, aliases),
            kind_char: *kind_char,
        },
        OpKind::IsConstant { value, kind_char } => OpKind::IsConstant {
            value: remap_value(value, aliases),
            kind_char: *kind_char,
        },
        OpKind::IsVirtual { value, kind_char } => OpKind::IsVirtual {
            value: remap_value(value, aliases),
            kind_char: *kind_char,
        },
        OpKind::IsInstance {
            obj,
            class_carrier,
            result_ty,
        } => OpKind::IsInstance {
            obj: remap_value(obj, aliases),
            class_carrier: remap_value(class_carrier, aliases),
            result_ty: result_ty.clone(),
        },
        OpKind::CallElidable {
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => {
            let remap_var = |var: &crate::flowspace::model::Variable| remap_value(var, aliases);
            OpKind::CallElidable {
                funcptr: remap_call_funcptr(funcptr, aliases),
                descriptor: descriptor.clone(),
                args_i: args_i.iter().map(remap_var).collect(),
                args_r: args_r.iter().map(remap_var).collect(),
                args_f: args_f.iter().map(remap_var).collect(),
                result_kind: *result_kind,
            }
        }
        OpKind::CallResidual {
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
            indirect_targets,
        } => {
            let remap_var = |var: &crate::flowspace::model::Variable| remap_value(var, aliases);
            OpKind::CallResidual {
                funcptr: remap_call_funcptr(funcptr, aliases),
                descriptor: descriptor.clone(),
                args_i: args_i.iter().map(remap_var).collect(),
                args_r: args_r.iter().map(remap_var).collect(),
                args_f: args_f.iter().map(remap_var).collect(),
                indirect_targets: indirect_targets.clone(),
                result_kind: *result_kind,
            }
        }
        OpKind::CallMayForce {
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => {
            let remap_var = |var: &crate::flowspace::model::Variable| remap_value(var, aliases);
            OpKind::CallMayForce {
                funcptr: remap_call_funcptr(funcptr, aliases),
                descriptor: descriptor.clone(),
                args_i: args_i.iter().map(remap_var).collect(),
                args_r: args_r.iter().map(remap_var).collect(),
                args_f: args_f.iter().map(remap_var).collect(),
                result_kind: *result_kind,
            }
        }
        OpKind::InlineCall {
            jitcode,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => {
            let remap_var = |var: &crate::flowspace::model::Variable| remap_value(var, aliases);
            OpKind::InlineCall {
                jitcode: jitcode.clone(),
                args_i: args_i.iter().map(remap_var).collect(),
                args_r: args_r.iter().map(remap_var).collect(),
                args_f: args_f.iter().map(remap_var).collect(),
                result_kind: *result_kind,
            }
        }
        OpKind::RecursiveCall {
            jd_index,
            greens_i,
            greens_r,
            greens_f,
            reds_i,
            reds_r,
            reds_f,
            result_kind,
        } => {
            let remap_var = |var: &crate::flowspace::model::Variable| remap_value(var, aliases);
            OpKind::RecursiveCall {
                jd_index: *jd_index,
                greens_i: greens_i.iter().map(remap_var).collect(),
                greens_r: greens_r.iter().map(remap_var).collect(),
                greens_f: greens_f.iter().map(remap_var).collect(),
                reds_i: reds_i.iter().map(remap_var).collect(),
                reds_r: reds_r.iter().map(remap_var).collect(),
                reds_f: reds_f.iter().map(remap_var).collect(),
                result_kind: *result_kind,
            }
        }
        OpKind::ConditionalCall {
            condition,
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
        } => {
            let remap_var = |var: &crate::flowspace::model::Variable| remap_value(var, aliases);
            OpKind::ConditionalCall {
                condition: remap_var(condition),
                funcptr: funcptr.clone(),
                descriptor: descriptor.clone(),
                args_i: args_i.iter().map(remap_var).collect(),
                args_r: args_r.iter().map(remap_var).collect(),
                args_f: args_f.iter().map(remap_var).collect(),
            }
        }
        OpKind::ConditionalCallValue {
            value,
            funcptr,
            descriptor,
            args_i,
            args_r,
            args_f,
            result_kind,
        } => {
            let remap_var = |var: &crate::flowspace::model::Variable| remap_value(var, aliases);
            OpKind::ConditionalCallValue {
                value: remap_var(value),
                funcptr: funcptr.clone(),
                descriptor: descriptor.clone(),
                args_i: args_i.iter().map(remap_var).collect(),
                args_r: args_r.iter().map(remap_var).collect(),
                args_f: args_f.iter().map(remap_var).collect(),
                result_kind: *result_kind,
            }
        }
    };
    SpaceOperation {
        result: op.result.clone(),
        kind,
    }
}

/// Rewrite `we_are_jitted()` calls to the `_we_are_jitted` symbolic
/// constant (`OpKind::ConstSymbolic`) on the model graph — the
/// JIT-codewriter counterpart of RPython's rtyper `specialize_call`
/// rewrite of the `direct_call` to `inputconst(Signed, _we_are_jitted)`
/// (`rpython/rlib/jit.py:403-406`).
///
/// pyre's rtyper types an ephemeral oracle and never rewrites the
/// surviving model graph, so the symbolic is injected here, by
/// `Transformer::transform`, after annotation (the annotator typed the
/// call `SomeBool` via the `we_are_jitted` `ExtRegistryEntry`) and
/// before per-op rewriting.  `rewrite_operation` then folds the
/// symbolic to `ConstBool(true)` keyed on the `SpecTag` identity,
/// mirroring `jtransform.py:1638 value is _we_are_jitted`.  Running on
/// the model graph rather than the annotated flowspace oracle keeps the
/// un-annotatable `SpecTag` out of `Bookkeeper.immutablevalue` (which
/// has no symbolic branch — in RPython the symbolic is likewise
/// introduced only post-annotation, at rtype).
fn fold_we_are_jitted_calls(graph: &mut crate::model::FunctionGraph) {
    for block in graph.blocks.iter_mut() {
        for op in block.operations.iter_mut() {
            let OpKind::Call {
                target: CallTarget::FunctionPath { segments },
                args,
                ..
            } = &op.kind
            else {
                continue;
            };
            if !args.is_empty()
                || segments.len() != 3
                || segments[0] != "majit_metainterp"
                || segments[1] != "jit"
                || segments[2] != "we_are_jitted"
            {
                continue;
            }
            op.kind = OpKind::ConstSymbolic {
                tag: crate::translator::backendopt::constfold::WE_ARE_JITTED_TAG_ID,
                ty: crate::model::ValueType::Bool,
            };
        }
    }
}

fn classify_hint_target(target: &CallTarget) -> Option<crate::hints::HintKind> {
    target
        .path_segments()
        .and_then(crate::hints::classify_hint_segments)
}

/// Match a `CallEffectOverride` pattern against a call target.
///
/// The match is loose only in the asymmetric receiver-root direction:
/// when the pattern leaves `receiver_root` as `None`, any target
/// receiver matches; otherwise both sides must agree, either by
/// exact source-syntax equality or — when the target carries a
/// `resolved_path` — by comparing the pattern against the path's
/// `impl_type_prefix()`, directly or via leaf-suffix `canonical_leaf`
/// (the `::`-joined path's trailing segment).
///
/// # Do not relax this into a leaf-name comparison
///
/// An override whose pattern never matches is **inert**: the call keeps the
/// effects the analysis derived for it. An override that matches the *wrong*
/// callee installs **another function's declared effects** on the call, which
/// the analysis then trusts. Inert is safe; wrong is a miscompile. The two are
/// one relaxation apart, and the relaxation is the obvious-looking repair when
/// a row reads 0 matches.
///
/// Measured against the registered graphs, matching on the trailing segment
/// alone would make an `RBigInt::sub` row also capture `time::Instant::sub`,
/// `core::ops::arith::<Impl>::sub` and `buffer::Buffer::sub`, and an
/// `RBigInt::hash` row capture 16 unrelated `hash` impls. Same failure class as
/// registering a callee family that cannot be proven closed.
///
/// A row that reads 0 is telling you its *spelling* is wrong, not that the
/// predicate is too strict. Every non-`Method` pattern is compared by full
/// structural equality below, so the pattern must carry the same segmentation
/// the resolver produces at the call site — see `CallEffectOverride`'s own docs
/// for how to obtain it.
fn call_target_matches_loose(pattern: &CallTarget, target: &CallTarget) -> bool {
    match (pattern, target) {
        (
            CallTarget::Method {
                name: pn,
                receiver_root: pr,
                ..
            },
            CallTarget::Method {
                name: tn,
                receiver_root: tr,
                resolved_path,
            },
        ) => {
            if pn != tn {
                return false;
            }
            match (pr.as_deref(), tr.as_deref()) {
                (Some(p), Some(t)) => {
                    if p == t {
                        return true;
                    }
                    if let Some(path) = resolved_path {
                        let prefix = path.impl_type_prefix();
                        if prefix == p {
                            return true;
                        }
                        if crate::parse::canonical_leaf(&prefix) == p {
                            return true;
                        }
                    }
                    false
                }
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
/// The `SoU.TO` discriminant of a `stroruni.*` oopspec's first argument:
/// `Ptr(STR)` (`rpy_string`), `Ptr(UNICODE)` (`rpy_unicode`), or
/// `Ptr(BYTEARRAY)` (`rpy_bytearray`). `Other` covers any non-string /
/// untyped operand.
enum StrOrUniKind {
    Str,
    Unicode,
    ByteArray,
    Other,
}

/// Classify a `stroruni.*` oopspec call's first argument by its pointee
/// struct, mirroring `_handle_stroruni_call`'s
/// `SoU.TO == rstr.STR / rstr.UNICODE / rbytearray.BYTEARRAY` chain
/// (`jtransform.py:2058-2076`). `STR` and `BYTEARRAY` both carry a `chars`
/// array of `Char`, so the struct name (`rpy_string` / `rpy_unicode` /
/// `rpy_bytearray`) is the discriminant, matching upstream's `SoU.TO`
/// identity comparison against the module-global structs.
fn stroruni_first_arg_kind(var: &crate::flowspace::model::Variable) -> StrOrUniKind {
    use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, PtrTarget};
    let Some(LowLevelType::Ptr(ptr)) = var.concretetype() else {
        return StrOrUniKind::Other;
    };
    let pointee = match ptr.TO {
        PtrTarget::ForwardReference(fwd) => match fwd.resolved() {
            Some(resolved) => resolved,
            None => return StrOrUniKind::Other,
        },
        other => LowLevelType::from(other),
    };
    let LowLevelType::Struct(s) = pointee else {
        return StrOrUniKind::Other;
    };
    match s._name.as_str() {
        "rpy_string" => StrOrUniKind::Str,
        "rpy_unicode" => StrOrUniKind::Unicode,
        "rpy_bytearray" => StrOrUniKind::ByteArray,
        _ => StrOrUniKind::Other,
    }
}

/// The result-layout fork `_handle_list_call`'s `"newlist_clear"` arm
/// selects between, mirroring upstream's `LIST = op.result.concretetype.TO;
/// resizable = isinstance(LIST, lltype.GcStruct)` split
/// (`jtransform.py:1762-1785`).  `item_ty` / `array_type_id` are recovered
/// from the result element lltype in the `Resized` / `Fixed` cases (C1: the
/// pre-fork arm hardcoded `Ref(None)` / `None`, so a non-pointer element
/// list was GC-traced as a pointer array and a resized result emitted a
/// bare array with no struct header); `Fallback` keeps the pre-fork shape
/// for an untyped-or-`OBJECTPTR` result (pending N7's typed-result tests).
enum NewlistClearShape {
    /// Result is `Ptr(GcStruct("list", {length, items}))` — the resized
    /// list header.  `do_resizable_newlist_clear` (jtransform.py:1938).
    Resized {
        item_ty: ValueType,
        array_type_id: Option<String>,
    },
    /// Result is `Ptr(Array(ITEM))` — a bare fixed-size cleared array.
    /// `do_fixed_newlist_clear` (jtransform.py:1858).
    Fixed {
        item_ty: ValueType,
        array_type_id: Option<String>,
    },
    /// No concretetype, or a pointee that is neither the `"list"` struct
    /// nor an array (a generic `OBJECTPTR` from an untyped test).
    Fallback,
}

/// Classify a `newlist_clear` result Variable by its pointee list layout,
/// recovering the items-array element `ValueType` (C1: never hardcoded)
/// from the result lltype.
///
/// The element lltype → `ValueType` path is `getkind` (`model::getkind`,
/// the canonical `LowLevelType → kind` projection) → `ConcreteType` →
/// `ValueType`, matching `Transformer::get_value_type`'s kind-collapsed
/// mapping (Char/Bool/Signed → `Int`, gc `Ptr` → `Ref`).
///
/// `array_type_id` is the ITEMS ARRAY identity string
/// (`cpu.arraydescrof(ARRAY)` cache key).  It is a Rust-type spelling
/// produced by the MIR front-end and is not recoverable from an rtyped
/// lltype here, so it is `None`; `arraydescrof(item_ty, None, Some(0), cc)`
/// derives a correct-width descr from `item_ty` alone — both the
/// `CallControl` path (`arraydescrof_concrete`'s `elem_ref == None` branch)
/// and the codewriter-less `assembler::arraydescrof` fallback key itemsize
/// and pointer-ness off `item_ty` when `array_type_id` is absent (pointer
/// elements stride by the target word, int/float by 8; the flag follows the
/// item type).
fn newlist_clear_shape(result: Option<&crate::flowspace::model::Variable>) -> NewlistClearShape {
    use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, PtrTarget};

    fn resolve(target: PtrTarget) -> Option<LowLevelType> {
        match target {
            PtrTarget::ForwardReference(fwd) => fwd.resolved(),
            other => Some(LowLevelType::from(other)),
        }
    }

    // Element `LowLevelType` → `ValueType`, collapsed through `getkind`'s
    // kind space.  `getkind` never returns `Void`/`Unknown` for a real
    // array element (it panics on the unsupported longlonglong / longfloat
    // / interiorptr shapes), so those arms are defensive and land on the
    // GC-pointer default.
    fn element_value_type(elem: &LowLevelType) -> ValueType {
        use crate::codewriter::type_state::ConcreteType;
        match crate::model::getkind(elem) {
            ConcreteType::Signed => ValueType::Int,
            ConcreteType::Float => ValueType::Float,
            ConcreteType::GcRef | ConcreteType::Void | ConcreteType::Unknown => {
                ValueType::Ref(None)
            }
        }
    }

    let Some(LowLevelType::Ptr(ptr)) = result.and_then(|v| v.concretetype()) else {
        return NewlistClearShape::Fallback;
    };
    let Some(pointee) = resolve(ptr.TO) else {
        return NewlistClearShape::Fallback;
    };
    match pointee {
        // Resized layout: the `"list"` GcStruct header.  Its `items` field
        // is `Ptr(GcArray(ITEM))`; walk struct → `items` → array → OF.
        LowLevelType::Struct(s) if s._name == "list" => {
            let Some(LowLevelType::Ptr(items_ptr)) = s._flds.get("items") else {
                return NewlistClearShape::Fallback;
            };
            let Some(LowLevelType::Array(a)) = resolve(items_ptr.TO.clone()) else {
                return NewlistClearShape::Fallback;
            };
            NewlistClearShape::Resized {
                item_ty: element_value_type(&a.OF),
                array_type_id: None,
            }
        }
        // Fixed layout: the pointee is the items array directly.
        LowLevelType::Array(a) => NewlistClearShape::Fixed {
            item_ty: element_value_type(&a.OF),
            array_type_id: None,
        },
        _ => NewlistClearShape::Fallback,
    }
}

fn map_user_oopspec_to_index(spec: &str) -> majit_ir::descr::OopSpecIndex {
    use majit_ir::descr::OopSpecIndex;
    // Normalize: `jit.isconstant(value)` → `jit.isconstant`
    let base = spec.split('(').next().unwrap_or(spec).trim();
    match base {
        // All jit.* oopspecs are intercepted by _handle_jit_call() before
        // reaching this function. Remaining oopspecs map to OS_* indices.
        "virtual_ref" | "virtual_ref_finish" => OopSpecIndex::JitForceVirtualizable,
        // jtransform.py:507-509: oopspec_name.endswith('dict.lookup')
        _ if base.ends_with("dict.lookup") => OopSpecIndex::DictLookup,
        _ => OopSpecIndex::None,
    }
}

/// Call sites matched, per `call_effects` override entry.
///
/// `call_target_matches_loose` reports a non-match by returning `false`, and
/// `classify_call` then falls through to the ordinary `describe_call` path.
/// So an entry that matches nothing behaves exactly like an entry that is
/// absent: the build is green, no test fails, and the effects row it was
/// written to publish is simply never attached. This counter is the only
/// channel that separates the two.
///
/// Keyed by the target's `Debug` rendering, never its `Display`: `Display`
/// joins a `FunctionPath`'s segments with `::`, and the segmentation is what
/// the match keys on — see
/// `function_path_override_matches_only_its_own_segmentation`.
///
/// `BTreeMap` so two runs of one program print the same bytes.
static OVERRIDE_MATCHES: Mutex<BTreeMap<String, usize>> = Mutex::new(BTreeMap::new());

/// The key `OVERRIDE_MATCHES` files a target under.
fn override_key(target: &CallTarget) -> String {
    format!("{target:?}")
}

/// Every `Method` target `classify_call` was actually invoked on, and the
/// total number of invocations.
///
/// `OVERRIDE_MATCHES` counts *successful* matches, so a zero there has two
/// causes it cannot tell apart: the predicate was consulted and said no, or
/// the call site never reached the predicate at all. The census's own
/// `method_shapes` table cannot settle it either — it walks every registered
/// graph, which is a different and much larger population than the one
/// `transform_graph` visits. This pair is the consumer-side denominator, so
/// "no such call site was ever classified" becomes falsifiable.
///
/// Gated on the census env var: `classify_call` is hot, and an unconditional
/// lock here would tax every build to answer a question nobody is asking.
static CLASSIFY_SEEN_METHODS: Mutex<BTreeMap<String, usize>> = Mutex::new(BTreeMap::new());
/// The same denominator for `FunctionPath`, keyed by segments.
///
/// Needed for the same reason: knowing an override's spelling differs from the
/// call site's is only half an explanation if that call site is never
/// classified either. Without this, a spelling fix reads as unblocking a row
/// that would stay inert regardless.
static CLASSIFY_SEEN_PATHS: Mutex<BTreeMap<String, usize>> = Mutex::new(BTreeMap::new());
static CLASSIFY_SEEN_TOTAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn record_classify_seen(target: &CallTarget) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("PYRE_CALLEE_CENSUS").is_some_and(|v| v == "1")) {
        return;
    }
    CLASSIFY_SEEN_TOTAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    match target {
        CallTarget::Method {
            name,
            receiver_root,
            ..
        } => {
            *CLASSIFY_SEEN_METHODS
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .entry(format!("name={name:?} receiver_root={receiver_root:?}"))
                .or_insert(0) += 1;
        }
        CallTarget::FunctionPath { segments, .. } => {
            *CLASSIFY_SEEN_PATHS
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .entry(format!("{segments:?}"))
                .or_insert(0) += 1;
        }
        _ => {}
    }
}

fn record_override_match(pattern: &CallTarget) {
    *OVERRIDE_MATCHES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .entry(override_key(pattern))
        .or_insert(0) += 1;
}

/// Every entry of `overrides` with the number of call sites it matched,
/// heaviest first, zero-match entries last.
///
/// The table is passed in rather than remembered, because the finding this
/// exists to surface is an entry with **no** matches — and an entry that
/// matched nothing leaves no trace in the counter to enumerate. Reading the
/// rows off the caller's table is what makes a `0` printable.
///
/// The header carries `entries` beside `matched`, so an empty list cannot be
/// read as a clean table when it is really a run where no override table was
/// installed at all.
pub fn call_effect_override_census(overrides: &[CallEffectOverride]) -> Vec<String> {
    let matches = OVERRIDE_MATCHES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut rows: Vec<(String, usize)> = overrides
        .iter()
        .map(|override_| {
            let key = override_key(&override_.target);
            let hits = matches.get(&key).copied().unwrap_or(0);
            (key, hits)
        })
        .collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    // A key counted but absent from this table means two different override
    // tables were installed in one process. That makes every count below a
    // partial figure, so it is reported rather than dropped.
    let listed: std::collections::BTreeSet<&String> = rows.iter().map(|(key, _)| key).collect();
    let unlisted: Vec<&String> = matches.keys().filter(|key| !listed.contains(key)).collect();
    let seen_methods = CLASSIFY_SEEN_METHODS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut lines = vec![
        format!(
            "call_effect_overrides entries={} matched={} inert={} sites={} unlisted={}",
            overrides.len(),
            rows.iter().filter(|(_, hits)| *hits > 0).count(),
            rows.iter().filter(|(_, hits)| *hits == 0).count(),
            matches.values().sum::<usize>(),
            unlisted.len(),
        ),
        // The denominator for every zero above: without it, "no entry
        // matched" and "the predicate was never consulted" print identically.
        format!(
            "classify_call_seen total={} method_sites={} method_shapes={}",
            CLASSIFY_SEEN_TOTAL.load(std::sync::atomic::Ordering::Relaxed),
            seen_methods.values().sum::<usize>(),
            seen_methods.len(),
        ),
    ];
    for (key, hits) in seen_methods.iter() {
        lines.push(format!("classify_call_method {hits:>6}  {key}"));
    }
    let seen_paths = CLASSIFY_SEEN_PATHS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    lines.push(format!(
        "classify_call_seen_paths sites={} shapes={}",
        seen_paths.values().sum::<usize>(),
        seen_paths.len(),
    ));
    for (key, hits) in seen_paths.iter() {
        lines.push(format!("classify_call_path {hits:>6}  {key}"));
    }
    lines.extend(rows.iter().map(|(key, hits)| {
        let verdict = if *hits == 0 { " INERT" } else { "" };
        format!("call_effect_override {hits:>7}x{verdict}  {key}")
    }));
    lines.extend(
        unlisted
            .iter()
            .map(|key| format!("call_effect_override UNLISTED  {key}")),
    );
    lines
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

    record_classify_seen(target);
    if let Some(override_) = overrides
        .iter()
        .find(|override_| call_target_matches_loose(&override_.target, target))
    {
        record_override_match(&override_.target);
        let descriptor = override_.descriptor.clone();
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
    use crate::codewriter::type_state::ConcreteType;
    use crate::model::{CallFuncPtr, CallTarget, FunctionGraph, LinkArg, OpKind, ValueType};

    #[test]
    fn integer_bounds_matches_rpython_helper() {
        assert_eq!(integer_bounds(1, true), (0, 256));
        assert_eq!(integer_bounds(1, false), (-128, 128));
        assert_eq!(integer_bounds(2, false), (-32768, 32768));
    }

    #[test]
    fn keep_operation_unchanged_returns_clone() {
        let config = GraphTransformConfig::default();
        let transformer = Transformer::new(&config);
        let op = SpaceOperation {
            result: None,
            kind: OpKind::Live,
        };
        assert_eq!(keep_operation_unchanged(&transformer, &op), op);
    }

    /// A `FunctionPath` override matches its own segmentation and nothing
    /// else, and a non-match is silent.
    ///
    /// `call_target_matches_loose` is loose only on the `Method` arm; a
    /// `FunctionPath` pattern falls through to `_ => pattern == target`,
    /// structural equality with no leaf-suffix tolerance. Both spellings
    /// below `Display` as `core::ptr::null`, and owner-segmentation records that pyre's
    /// two registration writers really do produce owner-split and
    /// owner-unsplit paths for one function. So an override table written
    /// from a `::`-joined listing is a guess about which split the call site
    /// carries, and guessing wrong costs a `false` and no diagnostic — the
    /// row is simply never consulted.
    #[test]
    fn function_path_override_matches_only_its_own_segmentation() {
        let split = CallTarget::FunctionPath {
            segments: vec!["core".into(), "ptr".into(), "null".into()],
        };
        let unsplit = CallTarget::FunctionPath {
            segments: vec!["core::ptr".into(), "null".into()],
        };
        assert_eq!(split.to_string(), unsplit.to_string());
        assert!(call_target_matches_loose(&split, &split));
        assert!(!call_target_matches_loose(&split, &unsplit));
        assert!(!call_target_matches_loose(&unsplit, &split));
    }

    /// A declared row reaches `EffectInfo` intact, and the `RandomEffects`
    /// pairing upstream forbids cannot be written.
    ///
    /// The row below is the one a primitive like `core::ptr::null` wants:
    /// elidable, cannot raise, allocates nothing. Upstream's constraint is
    /// `assert not (elidable_function and random_effects_on_gcobjs)`
    /// (`rffi.py:160`), and `random_effects_on_gcobjs` derives from
    /// "releases the GIL or runs a callback" — neither of which this does. So
    /// elidable-with-a-concrete-row is the orthodox state, not a liberty, and
    /// the type must keep it expressible while `DeclaredExtraEffect` having
    /// no `RandomEffects` member makes the forbidden pairing unspellable.
    #[test]
    fn declared_effects_reach_the_effect_info_intact() {
        let info = effect_info_for_kind(CallEffectKind::Declared(DeclaredCallEffects {
            extra: DeclaredExtraEffect::ElidableCannotRaise,
            can_collect: false,
            can_invalidate: false,
        }));
        assert_eq!(info.extraeffect, ExtraEffect::ElidableCannotRaise);
        assert!(info.check_is_elidable());
        assert!(!info.can_collect);
        assert!(!info.can_invalidate);
        // The default row says `can_collect: true`, so the assertion above is
        // reading the declaration and not the default.
        assert!(EffectInfo::default().can_collect);
    }

    /// The override census names an entry that matched nothing.
    ///
    /// Both entries below are well-formed and differ only in segmentation, so
    /// the build cannot tell them apart and neither can any existing test:
    /// one attaches its effects row, the other is consulted, declined, and
    /// forgotten. The census has to print the second as `INERT`, which is the
    /// whole reason it exists.
    ///
    /// Asserts on this test's own rows rather than the header totals —
    /// `OVERRIDE_MATCHES` is process-global and the lib tests run in
    /// parallel, so a total is not this test's to own. The segment names are
    /// deliberately unique so no other test can touch these two keys.
    #[test]
    fn override_census_names_an_entry_that_matched_nothing() {
        let live = CallTarget::FunctionPath {
            segments: vec!["__probe135".into(), "live".into(), "leaf".into()],
        };
        let inert = CallTarget::FunctionPath {
            segments: vec!["__probe135::inert".into(), "leaf".into()],
        };
        let overrides = vec![
            CallEffectOverride::new(live.clone(), CallEffectKind::Elidable),
            CallEffectOverride::new(inert.clone(), CallEffectKind::Elidable),
        ];
        assert!(classify_call(&live, &overrides).is_some());

        let lines = call_effect_override_census(&overrides);
        let row = |target: &CallTarget| {
            let key = override_key(target);
            lines
                .iter()
                .find(|line| line.ends_with(&key))
                .unwrap_or_else(|| panic!("census has no row for {key}"))
                .clone()
        };
        assert!(row(&live).contains("1x"), "{}", row(&live));
        assert!(!row(&live).contains("INERT"), "{}", row(&live));
        assert!(row(&inert).contains("0x INERT"), "{}", row(&inert));
    }

    /// the GotoIfNotOp lowering Stage 1: `int_lt(a, b); exitswitch = t` fuses into a
    /// `Fused { opname: "int_lt", args: [a, b] }` switch, the `int_lt`
    /// op is removed, and the `t` riding a link's args is replaced by
    /// that link's bool constant (`jtransform.py:196-234`).
    #[test]
    fn optimize_goto_if_not_fuses_int_lt_compare() {
        use crate::flowspace::model::ConstValue;
        use crate::model::{ExitCase, ExitSwitch, Link};
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;

        let mut graph = FunctionGraph::new("goto_if_not_fuse");
        let a = graph.alloc_value_var();
        let b = graph.alloc_value_var();
        // `t = int_lt(a, b)` — a Bool-producing comparison.
        let t = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "int_lt".to_string(),
                    lhs: a.clone(),
                    rhs: b.clone(),
                    result_ty: ValueType::Bool,
                },
                true,
            )
            .unwrap();
        // `v.concretetype == lltype.Bool` — stamp the raw Bool lltype
        // (`getkind`/`ConcreteType` would otherwise collapse it to int).
        t.set_concretetype(Some(LowLevelType::Bool));

        // Two destination blocks; the true arm carries `t` in its args.
        let if_false = graph.create_block();
        let if_true = graph.create_block();
        let true_iarg = graph.alloc_value_var();
        graph.push_inputarg_var(if_true, true_iarg);
        let false_link = Link::new_mixed(vec![], if_false, Some(ExitCase::Bool(false)));
        let true_link = Link::new_mixed(
            vec![LinkArg::Value(t.clone())],
            if_true,
            Some(ExitCase::Bool(true)),
        );
        let start = graph.startblock.0;
        graph.set_control_flow_metadata(
            graph.startblock,
            Some(ExitSwitch::Value(t.clone())),
            vec![false_link, true_link],
        );

        let fused = super::optimize_goto_if_not(&mut graph, start);
        assert!(fused, "supported compare must fuse");

        let block = &graph.blocks[start];
        // The `int_lt` op was removed.
        assert!(
            !block
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "int_lt")),
            "fused int_lt op must be removed"
        );
        // exitswitch == Fused { opname: "int_lt", args: [a, b] }.
        match &block.exitswitch {
            Some(ExitSwitch::Fused { opname, args }) => {
                assert_eq!(opname, "int_lt");
                assert_eq!(args.len(), 2);
                assert_eq!(args[0], a);
                assert_eq!(args[1], b);
            }
            other => panic!("expected Fused exitswitch, got {other:?}"),
        }
        // The `t` riding the true arm became a Bool const(true).
        let true_arm = &block.exits[1];
        assert!(
            matches!(
                &true_arm.args[0],
                LinkArg::Const(c) if c.value == ConstValue::Bool(true)
            ),
            "escaping v must be replaced with the link's bool constant, got {:?}",
            true_arm.args[0],
        );
    }

    /// the GotoIfNotOp lowering Stage 1: a non-supported result op (`int_add`) is NOT
    /// fusable — `optimize_goto_if_not` returns false and leaves the
    /// block untouched (`jtransform.py:206-209` opname gate).
    #[test]
    fn optimize_goto_if_not_rejects_unsupported_result_op() {
        use crate::model::{ExitCase, ExitSwitch, Link};
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;

        let mut graph = FunctionGraph::new("goto_if_not_reject");
        let a = graph.alloc_value_var();
        let b = graph.alloc_value_var();
        let t = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "int_add".to_string(),
                    lhs: a,
                    rhs: b,
                    result_ty: ValueType::Bool,
                },
                true,
            )
            .unwrap();
        t.set_concretetype(Some(LowLevelType::Bool));

        let if_false = graph.create_block();
        let if_true = graph.create_block();
        let false_link = Link::new_mixed(vec![], if_false, Some(ExitCase::Bool(false)));
        let true_link = Link::new_mixed(vec![], if_true, Some(ExitCase::Bool(true)));
        graph.set_control_flow_metadata(
            graph.startblock,
            Some(ExitSwitch::Value(t.clone())),
            vec![false_link, true_link],
        );

        let start = graph.startblock.0;
        let before = graph.blocks[start].clone();
        let fused = super::optimize_goto_if_not(&mut graph, start);
        assert!(!fused, "unsupported int_add result op must not fuse");
        // Block unchanged: op kept, exitswitch still Value(t).
        let after = &graph.blocks[start];
        assert_eq!(after.operations.len(), before.operations.len());
        assert!(matches!(&after.exitswitch, Some(ExitSwitch::Value(_))));
    }

    #[test]
    fn transform_graph_canonicalizes_frontend_bitops() {
        let mut graph = FunctionGraph::new("bitops");
        let lhs_var = graph.alloc_value_var();
        let rhs_var = graph.alloc_value_var();
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "bitxor".to_string(),
                    lhs: lhs_var,
                    rhs: rhs_var,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var));

        let transformed = transform_graph(&graph, &GraphTransformConfig::default());
        match &transformed.graph.block(graph.startblock).operations[0].kind {
            OpKind::BinOp { op, .. } => assert_eq!(op, "xor"),
            other => panic!("expected canonical BinOp, got {other:?}"),
        }
    }

    #[test]
    fn transform_graph_removes_same_as_and_remaps_return() {
        let mut graph = FunctionGraph::new("same_as_identity");
        let input_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "x".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let alias_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::UnaryOp {
                    op: "same_as".into(),
                    operand: input_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(alias_var));

        let transformed = transform_graph(&graph, &GraphTransformConfig::default());
        let block = transformed.graph.block(graph.startblock);

        assert!(
            !block.operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::UnaryOp { op, .. } if op == "same_as"
            )),
            "same_as must be eliminated before assembly: {:?}",
            block.operations
        );
        assert_eq!(block.exits.len(), 1);
        assert_eq!(block.exits[0].target, transformed.graph.returnblock);
        assert_eq!(block.exits[0].args, vec![LinkArg::Value(input_var)]);
    }

    #[test]
    fn transform_graph_remaps_guard_value_after_same_as_identity() {
        let mut graph = FunctionGraph::new("same_as_then_guard");
        let input_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "x".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let alias_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::UnaryOp {
                    op: "same_as".into(),
                    operand: input_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            graph.startblock,
            OpKind::GuardValue {
                value: alias_var,
                kind_char: 'i',
            },
            false,
        );
        graph.set_return(graph.startblock, None);

        let transformed = transform_graph(&graph, &GraphTransformConfig::default());
        let guard = transformed
            .graph
            .block(transformed.graph.startblock)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::GuardValue { value, .. } => Some(value),
                _ => None,
            })
            .expect("GuardValue must survive the transform");

        assert_eq!(
            guard, &input_var,
            "GuardValue.value must follow _do_renaming through same_as"
        );
    }

    #[test]
    fn transform_graph_remaps_vtable_method_receiver_after_same_as_identity() {
        let mut graph = FunctionGraph::new("same_as_then_vtable_method");
        let receiver_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "receiver".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let alias_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::UnaryOp {
                    op: "same_as".into(),
                    operand: receiver_var.clone(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            graph.startblock,
            OpKind::VtableMethodPtr {
                receiver: alias_var,
                trait_root: "Handler".into(),
                method_name: "run".into(),
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        let transformed = transform_graph(&graph, &GraphTransformConfig::default());
        let vtable_receiver = transformed
            .graph
            .block(transformed.graph.startblock)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::VtableMethodPtr { receiver, .. } => Some(receiver),
                _ => None,
            })
            .expect("VtableMethodPtr must survive the transform");

        assert_eq!(
            vtable_receiver, &receiver_var,
            "VtableMethodPtr.receiver must follow _do_renaming through same_as"
        );
    }

    #[test]
    fn transform_graph_coerces_mixed_float_add() {
        let mut graph = FunctionGraph::new("mixed_float_add");
        let lhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "lhs".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let rhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "rhs".into(),
                    ty: ValueType::Float,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs: lhs_var.clone(),
                    rhs: rhs_var.clone(),
                    result_ty: ValueType::Float,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&lhs_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&rhs_var, ConcreteType::Float);
        FunctionGraph::set_concretetype_of_inline(&result_var, ConcreteType::Float);

        let config = GraphTransformConfig::default();
        let transformed = Transformer::new(&config).transform(&graph);
        let ops = &transformed.graph.block(graph.startblock).operations;
        let cast_result = match &ops[2].kind {
            OpKind::UnaryOp {
                op,
                operand,
                result_ty,
            } => {
                assert_eq!(op, "cast_int_to_float");
                assert_eq!(*operand, lhs_var);
                assert_eq!(*result_ty, ValueType::Float);
                ops[2].result.clone().unwrap()
            }
            other => panic!("expected cast_int_to_float, got {other:?}"),
        };
        let cast_result_var = cast_result;
        match &ops[3].kind {
            OpKind::BinOp {
                op,
                lhs: rewritten_lhs,
                rhs: rewritten_rhs,
                result_ty,
            } => {
                assert_eq!(op, "float_add");
                assert_eq!(*rewritten_lhs, cast_result_var);
                assert_eq!(*rewritten_rhs, rhs_var);
                assert_eq!(*result_ty, ValueType::Float);
            }
            other => panic!("expected float_add, got {other:?}"),
        }
    }

    #[test]
    fn transform_graph_coerces_mixed_float_comparison() {
        let mut graph = FunctionGraph::new("mixed_float_eq");
        let lhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "lhs".into(),
                    ty: ValueType::Float,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let rhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "rhs".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "eq".into(),
                    lhs: lhs_var.clone(),
                    rhs: rhs_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&lhs_var, ConcreteType::Float);
        FunctionGraph::set_concretetype_of_inline(&rhs_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&result_var, ConcreteType::Signed);

        let config = GraphTransformConfig::default();
        let transformed = Transformer::new(&config).transform(&graph);
        let ops = &transformed.graph.block(graph.startblock).operations;
        let cast_result = match &ops[2].kind {
            OpKind::UnaryOp {
                op,
                operand,
                result_ty,
            } => {
                assert_eq!(op, "cast_int_to_float");
                assert_eq!(*operand, rhs_var);
                assert_eq!(*result_ty, ValueType::Float);
                ops[2].result.clone().unwrap()
            }
            other => panic!("expected cast_int_to_float, got {other:?}"),
        };
        let cast_result_var = cast_result;
        match &ops[3].kind {
            OpKind::BinOp {
                op,
                lhs: rewritten_lhs,
                rhs: rewritten_rhs,
                result_ty,
            } => {
                assert_eq!(op, "float_eq");
                assert_eq!(*rewritten_lhs, lhs_var);
                assert_eq!(*rewritten_rhs, cast_result_var);
                assert_eq!(*result_ty, ValueType::Int);
            }
            other => panic!("expected float_eq, got {other:?}"),
        }
    }

    #[test]
    fn transform_graph_lowers_float_mod_to_ll_math_fmod_residual_call() {
        let mut graph = FunctionGraph::new("float_mod");
        let lhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "lhs".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let rhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "rhs".into(),
                    ty: ValueType::Float,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "mod".into(),
                    lhs: lhs_var.clone(),
                    rhs: rhs_var.clone(),
                    result_ty: ValueType::Float,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&lhs_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&rhs_var, ConcreteType::Float);
        FunctionGraph::set_concretetype_of_inline(&result_var, ConcreteType::Float);

        let config = GraphTransformConfig::default();
        let transformed = Transformer::new(&config).transform(&graph);
        let ops = &transformed.graph.block(graph.startblock).operations;
        assert_eq!(ops.len(), 6, "Input + Input + cast + fnptr + call + Live");
        let cast_result = match &ops[2].kind {
            OpKind::UnaryOp {
                op,
                operand,
                result_ty,
            } => {
                assert_eq!(op, "cast_int_to_float");
                assert_eq!(*operand, lhs_var);
                assert_eq!(*result_ty, ValueType::Float);
                ops[2].result.clone().unwrap()
            }
            other => panic!("expected cast_int_to_float, got {other:?}"),
        };
        let expected_fnaddr =
            crate::call::symbolic_fnaddr_for_target(&CallTarget::function_path(["ll_math_fmod"]));
        assert!(matches!(&ops[3].kind, OpKind::ConstInt(fnaddr) if *fnaddr == expected_fnaddr));
        match &ops[4].kind {
            OpKind::CallResidual {
                funcptr,
                descriptor,
                args_i,
                args_r,
                args_f,
                result_kind,
                indirect_targets,
            } => {
                assert!(matches!(funcptr, CallFuncPtr::Value(_)));
                assert_eq!(
                    descriptor.extra_info.extraeffect,
                    ExtraEffect::ElidableCanRaise
                );
                assert!(args_i.is_empty());
                assert!(args_r.is_empty());
                let cast_result_var = cast_result;
                assert_eq!(args_f, &vec![cast_result_var, rhs_var]);
                assert_eq!(*result_kind, 'f');
                assert!(indirect_targets.is_none());
            }
            other => panic!("expected CallResidual, got {other:?}"),
        }
        assert!(matches!(ops[5].kind, OpKind::Live));
    }

    #[test]
    fn transform_graph_lowers_int_str_to_jit_int_str_residual_call() {
        let mut graph = FunctionGraph::new("int_str");
        let n_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "n".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::UnaryOp {
                    op: "str".into(),
                    operand: n_var.clone(),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&n_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&result_var, ConcreteType::GcRef);

        let config = GraphTransformConfig::default();
        let transformed = Transformer::new(&config).transform(&graph);
        let ops = &transformed.graph.block(graph.startblock).operations;
        assert_eq!(ops.len(), 4, "Input + fnptr + call + Live");
        let expected_fnaddr =
            crate::call::symbolic_fnaddr_for_target(&CallTarget::function_path(["jit_int_str"]));
        assert!(matches!(&ops[1].kind, OpKind::ConstInt(fnaddr) if *fnaddr == expected_fnaddr));
        match &ops[2].kind {
            OpKind::CallResidual {
                funcptr,
                descriptor,
                args_i,
                args_r,
                args_f,
                result_kind,
                indirect_targets,
            } => {
                assert!(matches!(funcptr, CallFuncPtr::Value(_)));
                assert_eq!(
                    descriptor.extra_info.extraeffect,
                    ExtraEffect::ElidableCanRaise
                );
                assert_eq!(args_i, &vec![n_var.clone()]);
                assert!(args_r.is_empty());
                assert!(args_f.is_empty());
                assert_eq!(*result_kind, 'r');
                assert!(indirect_targets.is_none());
            }
            other => panic!("expected CallResidual, got {other:?}"),
        }
        assert!(matches!(ops[3].kind, OpKind::Live));
    }

    #[test]
    fn transform_graph_tags_vable_fields() {
        let mut graph = FunctionGraph::new("test");
        let base_var = graph.alloc_value_var();
        let base_var_held = base_var.clone();
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldRead {
                base: base_var,
                field: crate::model::FieldDescriptor::new("next_instr", Some("Frame".into())),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        let config = GraphTransformConfig {
            vable_fields: vec![VirtualizableFieldDescriptor::new(
                "next_instr",
                Some("Frame".into()),
                0,
            )],
            ..Default::default()
        };
        let result = transform_graph(&graph, &config);
        assert_eq!(result.vable_rewrites, 1);
        // Should be rewritten to `-live-` + VableFieldRead
        // (`jtransform.py:845-847`).
        let ops = &result.graph.block(graph.startblock).operations;
        assert!(
            matches!(ops[0].kind, OpKind::Live),
            "virtualizable read must be led by -live-, got {:?}",
            ops[0].kind
        );
        let rewritten_op = &ops[1];
        let OpKind::VableFieldRead {
            base: rewritten_base,
            field_index,
            ..
        } = &rewritten_op.kind
        else {
            panic!("expected VableFieldRead, got {:?}", rewritten_op.kind);
        };
        assert_eq!(*field_index, 0);
        assert_eq!(rewritten_base, &base_var_held);
    }

    #[test]
    fn transform_graph_tags_vable_arrays_with_explicit_base() {
        let mut graph = FunctionGraph::new("test");
        let base_var = graph.alloc_value_var();
        let index_var = graph.alloc_value_var();
        let base_var_held = base_var.clone();
        let array_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: base_var,
                    field: crate::model::FieldDescriptor::new(
                        "locals_stack_w",
                        Some("Frame".into()),
                    ),
                    ty: ValueType::Ref(None),
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            graph.startblock,
            OpKind::ArrayRead {
                base: array_var,
                index: index_var,
                item_ty: ValueType::Int,
                array_type_id: None,
                nolength: false,
                pure: false,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        let config = GraphTransformConfig {
            vable_arrays: vec![VirtualizableFieldDescriptor::new_with_arraydescr(
                "locals_stack_w",
                Some("Frame".into()),
                0,
                8,
                true,
            )],
            ..Default::default()
        };
        let result = transform_graph(&graph, &config);
        assert_eq!(result.vable_rewrites, 1);
        // `jtransform.py:764-767` — `-live-` leads the vable array read.
        let ops = &result.graph.block(graph.startblock).operations;
        assert!(
            matches!(ops[1].kind, OpKind::Live),
            "virtualizable array read must be led by -live-, got {:?}",
            ops[1].kind
        );
        let rewritten_op = &ops[2];
        let OpKind::VableArrayRead {
            base: rewritten_base,
            array_index,
            ..
        } = &rewritten_op.kind
        else {
            panic!(
                "expected VableArrayRead with explicit base, got {:?}",
                rewritten_op.kind
            );
        };
        assert_eq!(*array_index, 0);
        assert_eq!(rewritten_base, &base_var_held);
    }

    /// `jtransform.py:126-127` + `:145-168 _check_no_vable_array` — the
    /// array a virtualizable field read produced may not leave the block
    /// along a link argument; the block that would consume it has no
    /// `vable_array_vars` entry for it, so no lowering could ever happen.
    ///
    /// Drives the whole `transform_graph`, so it is evidence that the
    /// transform enforces the rule and not only that the predicate is
    /// correct.  This is the same graph as
    /// `fresh_virtualizable_suppresses_vable_array_tracking` minus the
    /// hint: there the array never becomes a virtualizable array and the
    /// link is fine, here it does and the link is not.
    ///
    /// The expectation names the escape route, not just the upstream
    /// header: all four call sites share one panic body, so without the
    /// route the message cannot say which one fired.
    #[test]
    #[should_panic(expected = "Escaped via: link argument")]
    fn vable_array_escaping_the_block_on_a_link_arg_is_rejected() {
        use crate::model::Link;

        let mut graph = FunctionGraph::new("vable_array_escape");
        let frame_var = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, frame_var.clone());
        let array_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: frame_var,
                    field: crate::model::FieldDescriptor::new(
                        "locals_stack_w",
                        Some("Frame".into()),
                    ),
                    ty: ValueType::Ref(None),
                    pure: false,
                },
                true,
            )
            .unwrap();

        // The consumer block receives the array as a link argument instead
        // of reading it itself, so its `vable_array_vars` — rebuilt per
        // block — has no entry to lower the access against.
        let consumer = graph.create_block();
        let phi = graph.alloc_value_var();
        graph.push_inputarg_var(consumer, phi.clone());
        graph.set_return(consumer, Some(phi));
        let link = Link::from_variables(&graph, vec![array_var], consumer, None);
        graph.set_control_flow_metadata(graph.startblock, None, vec![link]);

        let config = GraphTransformConfig {
            vable_arrays: vec![VirtualizableFieldDescriptor::new_with_arraydescr(
                "locals_stack_w",
                Some("Frame".into()),
                0,
                8,
                true,
            )],
            ..Default::default()
        };
        transform_graph(&graph, &config);
    }

    /// `jtransform.py:990-993` — a virtualizable the graph just allocated
    /// is not under the virtualizable protocol yet, so its array field
    /// read is an ordinary `getfield` and must not enter
    /// `vable_array_vars`.  This is the frame-construction shape
    /// `pypy/interpreter/pyframe.py:99 hint(self, access_directly=True,
    /// fresh_virtualizable=True)` marks: the array leaves the block, and
    /// without the suppression `_check_no_vable_array` would reject the
    /// graph.
    #[test]
    fn fresh_virtualizable_suppresses_vable_array_tracking() {
        use crate::model::Link;

        let mut graph = FunctionGraph::new("fresh_vable_array");
        let frame_var = graph.alloc_value_var();
        let hinted_var = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, frame_var.clone());
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["hint_fresh_virtualizable"]),
                args: vec![frame_var],
                result_ty: ValueType::Ref(None),
            },
            false,
        );
        graph
            .block_mut(graph.startblock)
            .operations
            .last_mut()
            .unwrap()
            .result = Some(hinted_var.clone());
        let array_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: hinted_var,
                    field: crate::model::FieldDescriptor::new(
                        "locals_stack_w",
                        Some("Frame".into()),
                    ),
                    ty: ValueType::Ref(None),
                    pure: false,
                },
                true,
            )
            .unwrap();

        // The same escape that `vable_array_escaping_the_block_on_a_link_arg_is_rejected`
        // rejects — here it must be accepted, because the array never
        // became a virtualizable array.
        let consumer = graph.create_block();
        let phi = graph.alloc_value_var();
        graph.push_inputarg_var(consumer, phi.clone());
        graph.set_return(consumer, Some(phi));
        let link = Link::from_variables(&graph, vec![array_var], consumer, None);
        graph.set_control_flow_metadata(graph.startblock, None, vec![link]);

        let config = GraphTransformConfig {
            vable_arrays: vec![VirtualizableFieldDescriptor::new_with_arraydescr(
                "locals_stack_w",
                Some("Frame".into()),
                0,
                8,
                true,
            )],
            ..Default::default()
        };
        let result = transform_graph(&graph, &config);
        assert_eq!(
            result.vable_rewrites, 0,
            "a fresh virtualizable must produce no virtualizable rewrite"
        );
        let ops = &result.graph.block(graph.startblock).operations;
        assert!(
            ops.iter()
                .any(|op| matches!(op.kind, OpKind::FieldRead { .. })),
            "the array field read must stay an ordinary getfield, got {ops:?}"
        );
    }

    /// Taking the **address** of a virtualizable array field is not a read
    /// of it, so it must not enter `vable_array_vars`.
    ///
    /// The front models a reference as an alias of its referent, so
    /// `&raw mut (*frame).locals_cells_stack_w` arrives here as a
    /// `FieldRead`.  Tracking it would drop the op — the array is only
    /// materialised at an array access — and leave the consumer that
    /// wanted the address with an undefined operand.
    /// `pyframe::FrameLocalsRoot::new` is the shape: it hands the slot
    /// address to `try_gc_add_root`.
    #[test]
    fn address_of_a_vable_array_field_is_not_tracked() {
        use crate::model::Link;

        let mut graph = FunctionGraph::new("vable_array_addr_of");
        let frame_var = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, frame_var.clone());
        let slot_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: frame_var,
                    field: crate::model::FieldDescriptor::new(
                        "locals_stack_w",
                        Some("Frame".into()),
                    )
                    .with_base_is_deref(true)
                    .with_taken_by_address(true),
                    ty: ValueType::Ref(None),
                    pure: false,
                },
                true,
            )
            .unwrap();

        // The address leaves the block, exactly as it does on the way to
        // `try_gc_add_root`.  Were the array tracked, this link would trip
        // `_check_no_vable_array`.
        let consumer = graph.create_block();
        let phi = graph.alloc_value_var();
        graph.push_inputarg_var(consumer, phi.clone());
        graph.set_return(consumer, Some(phi));
        let link = Link::from_variables(&graph, vec![slot_var], consumer, None);
        graph.set_control_flow_metadata(graph.startblock, None, vec![link]);

        let config = GraphTransformConfig {
            vable_arrays: vec![VirtualizableFieldDescriptor::new_with_arraydescr(
                "locals_stack_w",
                Some("Frame".into()),
                0,
                8,
                true,
            )],
            ..Default::default()
        };
        let result = transform_graph(&graph, &config);
        assert_eq!(
            result.vable_rewrites, 0,
            "an address-of must produce no virtualizable rewrite"
        );
        let ops = &result.graph.block(graph.startblock).operations;
        assert!(
            ops.iter()
                .any(|op| matches!(op.kind, OpKind::FieldRead { .. })),
            "the projection must stay an ordinary getfield, got {ops:?}"
        );
    }

    /// The scalar half of `jtransform.py:990-993`: a redirected field read
    /// on a just-allocated virtualizable stays a plain `getfield`.
    /// `access_directly` alone does NOT suppress it — see
    /// `transform_graph_consumes_identity_virtualizable_hints`, which
    /// asserts the lowering still fires for that hint.
    #[test]
    fn fresh_virtualizable_suppresses_vable_field_read() {
        let mut graph = FunctionGraph::new("fresh_vable_field");
        let frame_var = graph.alloc_value_var();
        let hinted_var = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, frame_var.clone());
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["hint_fresh_virtualizable"]),
                args: vec![frame_var],
                result_ty: ValueType::Ref(None),
            },
            false,
        );
        graph
            .block_mut(graph.startblock)
            .operations
            .last_mut()
            .unwrap()
            .result = Some(hinted_var.clone());
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldRead {
                base: hinted_var,
                field: crate::model::FieldDescriptor::new("next_instr", Some("Frame".into())),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        let result = transform_graph(
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

        assert_eq!(result.vable_rewrites, 0);
        let ops = &result.graph.block(graph.startblock).operations;
        assert!(
            !ops.iter()
                .any(|op| matches!(op.kind, OpKind::VableFieldRead { .. })),
            "a fresh virtualizable must not lower to VableFieldRead, got {ops:?}"
        );
    }

    /// Build a one-block graph reading `next_instr` off an inputarg whose
    /// descriptor records the given container shape, and lower it against a
    /// config that declares `next_instr` a virtualizable field.
    fn vable_read_with_base_shape(base_is_deref: Option<bool>) -> GraphTransformResult {
        let mut graph = FunctionGraph::new("base_shape");
        let frame_var = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, frame_var.clone());
        let mut field = crate::model::FieldDescriptor::new("next_instr", Some("Frame".into()));
        field.base_is_deref = base_is_deref;
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldRead {
                base: frame_var,
                field,
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        graph.set_return(graph.startblock, None);
        transform_graph(
            &graph,
            &GraphTransformConfig {
                vable_fields: vec![VirtualizableFieldDescriptor::new(
                    "next_instr",
                    Some("Frame".into()),
                    0,
                )],
                ..Default::default()
            },
        )
    }

    /// A field read on a non-dereferenced local aggregate cannot be an
    /// access to a live virtualizable, so it must not lower — the
    /// structural stand-in for the `fresh_virtualizable` hint that cannot
    /// reach the frame constructor's later blocks.
    ///
    /// The `Some(true)` and `None` arms are the control: without them a
    /// suppression that simply never fires would pass this test.
    #[test]
    fn non_dereferenced_base_suppresses_vable_field_read() {
        // Local aggregate — suppressed.
        let local = vable_read_with_base_shape(Some(false));
        assert_eq!(
            local.vable_rewrites, 0,
            "a non-dereferenced container must not lower to VableFieldRead"
        );
        assert!(
            !local
                .graph
                .block(local.graph.startblock)
                .operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::VableFieldRead { .. }))
        );

        // Reached through a pointer — this is a real virtualizable access
        // and MUST still lower.
        let through_ptr = vable_read_with_base_shape(Some(true));
        assert_eq!(
            through_ptr.vable_rewrites, 1,
            "a dereferenced container is a live virtualizable and must still lower"
        );

        // Producer did not record the shape — unchanged behaviour, so a
        // descriptor built anywhere else cannot silently lose the protocol.
        let unknown = vable_read_with_base_shape(None);
        assert_eq!(
            unknown.vable_rewrites, 1,
            "an unrecorded container shape must keep the pre-existing lowering"
        );
    }

    /// Build a one-block graph *writing* `next_instr` on an inputarg whose
    /// descriptor records the given container shape.
    fn vable_write_with_base_shape(base_is_deref: Option<bool>) -> GraphTransformResult {
        let mut graph = FunctionGraph::new("base_shape_write");
        let frame_var = graph.alloc_value_var();
        let value_var = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, frame_var.clone());
        graph.push_inputarg_var(graph.startblock, value_var.clone());
        let mut field = crate::model::FieldDescriptor::new("next_instr", Some("Frame".into()));
        field.base_is_deref = base_is_deref;
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldWrite {
                base: frame_var,
                field,
                value: LinkArg::Value(value_var),
                ty: ValueType::Int,
            },
            false,
        );
        graph.set_return(graph.startblock, None);
        transform_graph(
            &graph,
            &GraphTransformConfig {
                vable_fields: vec![VirtualizableFieldDescriptor::new(
                    "next_instr",
                    Some("Frame".into()),
                    0,
                )],
                ..Default::default()
            },
        )
    }

    /// The write arm of the structural suppression.
    ///
    /// This is the arm worth pinning hardest.  In the frame constructor the
    /// non-dereferenced sites are 12, and **5 of them are writes** — the
    /// stores that read the fields back out of the GC root bracket.  An
    /// unsuppressed `setfield_vable_*` against a stack aggregate that has
    /// no `vable_token` is a worse failure than an over-tracked read, and
    /// a read-only test would not see it regress.
    #[test]
    fn non_dereferenced_base_suppresses_vable_field_write() {
        let local = vable_write_with_base_shape(Some(false));
        assert_eq!(
            local.vable_rewrites, 0,
            "a write into a non-dereferenced container must not lower to VableFieldWrite"
        );
        assert!(
            !local
                .graph
                .block(local.graph.startblock)
                .operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::VableFieldWrite { .. })),
            "structural suppression must reach the write arm"
        );

        // Controls, so a suppression that never fires cannot pass.
        assert_eq!(
            vable_write_with_base_shape(Some(true)).vable_rewrites,
            1,
            "a write through a pointer is a live virtualizable write and must still lower"
        );
        assert_eq!(
            vable_write_with_base_shape(None).vable_rewrites,
            1,
            "an unrecorded container shape must keep the pre-existing lowering"
        );
    }

    /// `jtransform.py:923` routes the write side through the same
    /// `is_virtualizable_getset`, so the fresh-virtualizable arm
    /// suppresses `setfield_vable_*` too — this is the shape a
    /// constructor uses to initialise the frame it just allocated.
    #[test]
    fn fresh_virtualizable_suppresses_vable_field_write() {
        let mut graph = FunctionGraph::new("fresh_vable_write");
        let frame_var = graph.alloc_value_var();
        let hinted_var = graph.alloc_value_var();
        let value_var = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, frame_var.clone());
        graph.push_inputarg_var(graph.startblock, value_var.clone());
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["hint_fresh_virtualizable"]),
                args: vec![frame_var],
                result_ty: ValueType::Ref(None),
            },
            false,
        );
        graph
            .block_mut(graph.startblock)
            .operations
            .last_mut()
            .unwrap()
            .result = Some(hinted_var.clone());
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldWrite {
                base: hinted_var,
                field: crate::model::FieldDescriptor::new("next_instr", Some("Frame".into())),
                value: LinkArg::Value(value_var),
                ty: ValueType::Int,
            },
            false,
        );
        graph.set_return(graph.startblock, None);

        let result = transform_graph(
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

        assert_eq!(result.vable_rewrites, 0);
        let ops = &result.graph.block(graph.startblock).operations;
        assert!(
            !ops.iter()
                .any(|op| matches!(op.kind, OpKind::VableFieldWrite { .. })),
            "a fresh virtualizable must not lower to VableFieldWrite, got {ops:?}"
        );
    }

    #[test]
    fn transform_graph_requires_matching_field_owner_root() {
        let mut graph = FunctionGraph::new("test");
        let base_var = graph.alloc_value_var();
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldRead {
                base: base_var,
                field: crate::model::FieldDescriptor::new("next_instr", Some("OtherFrame".into())),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        let config = GraphTransformConfig {
            vable_fields: vec![VirtualizableFieldDescriptor::new(
                "next_instr",
                Some("Frame".into()),
                0,
            )],
            ..Default::default()
        };
        let result = transform_graph(&graph, &config);
        assert_eq!(result.vable_rewrites, 0);
        assert!(matches!(
            result.graph.block(graph.startblock).operations[0].kind,
            OpKind::FieldRead { .. }
        ));
    }

    #[test]
    fn transform_graph_types_fieldwrite_from_value_kind() {
        let mut graph = FunctionGraph::new("test");
        let base_var = graph.alloc_value_var();
        let value_var = graph.alloc_value_var();
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldWrite {
                base: base_var,
                field: crate::model::FieldDescriptor::new("x", Some("Point".into())),
                value: crate::model::LinkArg::Value(value_var.clone()),
                ty: ValueType::Unknown,
            },
            false,
        );
        graph.set_return(graph.startblock, None);
        FunctionGraph::set_concretetype_of_inline(
            &value_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );

        let config = GraphTransformConfig::default();
        let result = Transformer::new(&config).transform(&graph);

        match &result.graph.block(graph.startblock).operations[0].kind {
            OpKind::FieldWrite { ty, .. } => assert_eq!(*ty, ValueType::Int),
            other => panic!("expected FieldWrite, got {other:?}"),
        }
    }

    #[test]
    fn transform_graph_types_arraywrite_from_value_kind() {
        let mut graph = FunctionGraph::new("test");
        let base_var = graph.alloc_value_var();
        let index_var = graph.alloc_value_var();
        let value_var = graph.alloc_value_var();
        graph.push_op_var(
            graph.startblock,
            OpKind::ArrayWrite {
                base: base_var,
                index: index_var,
                value: crate::model::LinkArg::Value(value_var.clone()),
                item_ty: ValueType::Unknown,
                array_type_id: None,
                nolength: false,
            },
            false,
        );
        graph.set_return(graph.startblock, None);
        FunctionGraph::set_concretetype_of_inline(
            &value_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );

        let config = GraphTransformConfig::default();
        let result = Transformer::new(&config).transform(&graph);

        match &result.graph.block(graph.startblock).operations[0].kind {
            OpKind::ArrayWrite { item_ty, .. } => assert_eq!(*item_ty, ValueType::Int),
            other => panic!("expected ArrayWrite, got {other:?}"),
        }
    }

    #[test]
    fn transform_graph_types_fieldread_from_result_kind() {
        let mut graph = FunctionGraph::new("test");
        let base_var = graph.alloc_value_var();
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: base_var,
                    field: crate::model::FieldDescriptor::new("x", Some("Point".into())),
                    ty: ValueType::Unknown,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));
        FunctionGraph::set_concretetype_of_inline(
            &result_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );

        let config = GraphTransformConfig::default();
        let result = Transformer::new(&config).transform(&graph);

        match &result.graph.block(graph.startblock).operations[0].kind {
            OpKind::FieldRead { ty, .. } => assert_eq!(*ty, ValueType::Int),
            other => panic!("expected FieldRead, got {other:?}"),
        }
    }

    #[test]
    fn transform_graph_types_arrayread_from_result_kind() {
        let mut graph = FunctionGraph::new("test");
        let base_var = graph.alloc_value_var();
        let index_var = graph.alloc_value_var();
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::ArrayRead {
                    base: base_var,
                    index: index_var,
                    item_ty: ValueType::Unknown,
                    array_type_id: None,
                    nolength: false,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));
        FunctionGraph::set_concretetype_of_inline(
            &result_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );

        let config = GraphTransformConfig::default();
        let result = Transformer::new(&config).transform(&graph);

        match &result.graph.block(graph.startblock).operations[0].kind {
            OpKind::ArrayRead { item_ty, .. } => assert_eq!(*item_ty, ValueType::Int),
            other => panic!("expected ArrayRead, got {other:?}"),
        }
    }

    #[test]
    fn transform_graph_classifies_calls() {
        let mut graph = FunctionGraph::new("test");
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::method("call_callable", Some("PyFrame".into())),
                args: vec![],
                result_ty: ValueType::Ref(None),
            },
            false,
        );
        graph.set_return(graph.startblock, None);

        let result = transform_graph(
            &graph,
            &crate::test_support::pyre_pipeline_config().transform,
        );
        assert_eq!(result.calls_classified, 1);
        assert!(matches!(
            result.graph.block(graph.startblock).operations[0].kind,
            OpKind::ConstInt(_)
        ));
        assert!(matches!(
            result.graph.block(graph.startblock).operations[1].kind,
            OpKind::CallResidual { .. }
        ));
    }

    #[test]
    fn residual_call_unknown_result_uses_resolved_type_for_opcode_and_descr() {
        let mut graph = FunctionGraph::new("outer");
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Call {
                    target: CallTarget::function_path(["unknown_external_int"]),
                    args: vec![],
                    result_ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&result_var, ConcreteType::Signed);

        let mut cc = crate::call::CallControl::new();
        let config = GraphTransformConfig::default();
        let transformed = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .transform(&graph);

        let (result_kind, descriptor) = transformed
            .graph
            .block(graph.startblock)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::CallResidual {
                    result_kind,
                    descriptor,
                    ..
                } => Some((*result_kind, descriptor)),
                _ => None,
            })
            .expect("residual call must be emitted");

        assert_eq!(result_kind, 'i');
        assert_eq!(descriptor.result_ir_type(), majit_ir::Type::Int);
    }

    #[test]
    fn int_mod_unknown_ast_result_rewrites_when_type_state_proves_signed() {
        let mut graph = FunctionGraph::new("int_mod_body");
        let lhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "lhs".into(),
                    ty: ValueType::Unknown,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let rhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "rhs".into(),
                    ty: ValueType::Unknown,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "mod".to_string(),
                    lhs: lhs_var.clone(),
                    rhs: rhs_var.clone(),
                    result_ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&lhs_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&rhs_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&result_var, ConcreteType::Signed);
        let config = GraphTransformConfig::default();
        let mut cc = crate::call::CallControl::new();
        let transformed = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .transform(&graph);

        let ops = &transformed.graph.block(graph.startblock).operations;
        assert!(
            !ops.iter()
                .any(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "mod")),
            "bare int_mod must not survive jtransform: {ops:?}"
        );
        let residual = ops
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::CallResidual {
                    descriptor,
                    result_kind,
                    args_i,
                    ..
                } => Some((descriptor, *result_kind, args_i)),
                _ => None,
            })
            .expect("mod must rewrite to residual helper call");
        assert_eq!(residual.1, 'i');
        assert_eq!(residual.2, &vec![lhs_var, rhs_var]);
        assert_eq!(residual.0.result_ir_type(), majit_ir::Type::Int);
        assert_eq!(residual.0.get_extra_info().oopspecindex, OopSpecIndex::None);
    }

    #[test]
    fn mod_assign_rewrites_directly_to_int_mod_residual() {
        // Rust low-level → RPython low-level: pyre constructs
        // `mod_assign` from Rust's `%=` operator on i64, which has
        // C-truncating remainder semantics.  That maps to RPython's
        // explicit `llop.int_mod` route (`support.py:266-271
        // _ll_2_int_mod`), not to Python-level `%=` / `rtype_mod`
        // (`rint.py:260-262`, which calls `py_mod`).  The assign arm
        // must emit the `_ll_2_int_mod` residual directly because
        // `transform_op`'s Replace output is not re-traversed within
        // the same pass — leaving a bare `mod` here would leak past
        // the jtransform rewrite gate.
        let mut graph = FunctionGraph::new("mod_assign_body");
        let lhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "lhs".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let rhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "rhs".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "mod_assign".to_string(),
                    lhs: lhs_var.clone(),
                    rhs: rhs_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&lhs_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&rhs_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&result_var, ConcreteType::Signed);
        let config = GraphTransformConfig::default();
        let mut cc = crate::call::CallControl::new();
        let transformed = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .transform(&graph);

        let ops = &transformed.graph.block(graph.startblock).operations;
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::BinOp { op, .. } if op == "mod_assign" || op == "mod"
            )),
            "mod_assign must not survive as a bare BinOp: {ops:?}"
        );
        let residual = ops
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::CallResidual {
                    descriptor,
                    result_kind,
                    args_i,
                    ..
                } => Some((descriptor, *result_kind, args_i)),
                _ => None,
            })
            .expect("mod_assign must rewrite to residual _ll_2_int_mod call");
        assert_eq!(residual.1, 'i');
        assert_eq!(residual.2, &vec![lhs_var, rhs_var]);
        assert_eq!(residual.0.result_ir_type(), majit_ir::Type::Int);
    }

    #[test]
    fn div_assign_rewrites_directly_to_int_floordiv_residual() {
        // Rust low-level → RPython low-level: pyre constructs
        // `div_assign` from Rust's `/=` operator on i64, which has
        // C-truncating division semantics.  That maps to RPython's
        // explicit `llop.int_floordiv` route (`support.py:255-264
        // _ll_2_int_floordiv`), not to Python-level `/=` /
        // `rtype_inplace_div` (`rint.py:253-255`, which aliases to
        // `rtype_floordiv` and calls `py_div`).  The assign arm
        // aliases `"div"` to `floordiv` and emits the residual
        // directly so no bare `div` / `floordiv` survives this pass.
        let mut graph = FunctionGraph::new("div_assign_body");
        let lhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "lhs".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let rhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "rhs".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "div_assign".to_string(),
                    lhs: lhs_var.clone(),
                    rhs: rhs_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&lhs_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&rhs_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&result_var, ConcreteType::Signed);
        let config = GraphTransformConfig::default();
        let mut cc = crate::call::CallControl::new();
        let transformed = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .transform(&graph);

        let ops = &transformed.graph.block(graph.startblock).operations;
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::BinOp { op, .. }
                    if op == "div_assign" || op == "div" || op == "floordiv"
            )),
            "div_assign must not survive as a bare BinOp: {ops:?}"
        );
        let residual = ops
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::CallResidual {
                    descriptor,
                    result_kind,
                    args_i,
                    ..
                } => Some((descriptor, *result_kind, args_i)),
                _ => None,
            })
            .expect("div_assign must rewrite to residual _ll_2_int_floordiv call");
        assert_eq!(residual.1, 'i');
        assert_eq!(residual.2, &vec![lhs_var, rhs_var]);
        assert_eq!(residual.0.result_ir_type(), majit_ir::Type::Int);
    }

    #[test]
    fn plain_int_div_rewrites_directly_to_int_floordiv_residual() {
        // `rint.py:253-255 rtype_div = rtype_floordiv`: a plain
        // `BinOp { op:"div" }` over int operands (Rust `a / b` on
        // i64s) routes through the same `_ll_2_int_floordiv`
        // residual as `floordiv`.  RPython has no `int_div` op; the
        // rtyper aliases `div` to `floordiv` for integer reprs.
        let mut graph = FunctionGraph::new("int_div_body");
        let lhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "lhs".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let rhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "rhs".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "div".to_string(),
                    lhs: lhs_var.clone(),
                    rhs: rhs_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&lhs_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&rhs_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&result_var, ConcreteType::Signed);
        let config = GraphTransformConfig::default();
        let mut cc = crate::call::CallControl::new();
        let transformed = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .transform(&graph);

        let ops = &transformed.graph.block(graph.startblock).operations;
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::BinOp { op, .. } if op == "div" || op == "floordiv"
            )),
            "plain int div must not survive as a bare BinOp: {ops:?}"
        );
        let residual = ops
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::CallResidual {
                    descriptor,
                    result_kind,
                    args_i,
                    ..
                } => Some((descriptor, *result_kind, args_i)),
                _ => None,
            })
            .expect("int div must rewrite to residual _ll_2_int_floordiv call");
        assert_eq!(residual.1, 'i');
        assert_eq!(residual.2, &vec![lhs_var, rhs_var]);
        assert_eq!(residual.0.result_ir_type(), majit_ir::Type::Int);
    }

    #[test]
    fn residual_direct_call_materializes_funcptr_const() {
        let mut cc = crate::call::CallControl::new();
        let target = CallTarget::function_path(["custom_reader"]);
        let mut graph = FunctionGraph::new("test");
        let arg_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "arg".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: target.clone(),
                args: vec![arg_var],
                result_ty: ValueType::Ref(None),
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
        let result = transformer.transform(&graph);
        let ops = &result.graph.block(graph.startblock).operations;
        assert_eq!(ops.len(), 4, "Input + ConstInt + CallResidual + Live");
        match &ops[1].kind {
            OpKind::ConstInt(fnaddr) => {
                assert_eq!(*fnaddr, cc.fnaddr_for_target(&target));
            }
            other => panic!("expected materialized funcptr ConstInt, got {other:?}"),
        }
        match &ops[2].kind {
            OpKind::CallResidual {
                funcptr: CallFuncPtr::Value(_),
                ..
            } => {}
            other => panic!("expected CallResidual with runtime funcptr, got {other:?}"),
        }
        assert!(matches!(ops[3].kind, OpKind::Live));
    }

    #[test]
    fn float_of_unsigned_projects_to_registered_cast_helper() {
        let mut cc = crate::call::CallControl::new();
        cc.register_macro_helper_trace_fnaddr("majit_metainterp::cast_uint_to_float", 0x1234);
        let mut graph = FunctionGraph::new("cast_uint_to_float_test");
        let arg = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "arg".into(),
                    ty: ValueType::Unsigned,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let result = graph
            .push_op_var(
                graph.startblock,
                OpKind::Call {
                    target: CallTarget::function_path(["float"]),
                    args: vec![arg],
                    result_ty: ValueType::Float,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result));

        let config = GraphTransformConfig::default();
        let transformed = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .transform(&graph);
        assert!(
            transformed
                .graph
                .block(graph.startblock)
                .operations
                .iter()
                .any(|op| matches!(op.kind, OpKind::ConstInt(0x1234)))
        );
    }

    #[test]
    fn transform_graph_uses_explicit_call_effect_overrides() {
        // RPython: residual calls always produce residual_call_*, regardless
        // of effect. The effect is only in the calldescr (descriptor).
        let mut graph = FunctionGraph::new("test");
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["custom_reader"]),
                args: vec![],
                result_ty: ValueType::Ref(None),
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        let result = transform_graph(
            &graph,
            &GraphTransformConfig {
                call_effects: vec![CallEffectOverride::new(
                    CallTarget::function_path(["custom_reader"]),
                    CallEffectKind::MayForce,
                )],
                ..Default::default()
            },
        );
        let ops = &result.graph.block(graph.startblock).operations;
        // RPython: always residual_call_*, effect in descriptor only.
        assert!(matches!(ops[0].kind, OpKind::ConstInt(_)));
        assert!(matches!(ops[1].kind, OpKind::CallResidual { .. }));
        // Verify the effect is correctly carried in the descriptor.
        if let OpKind::CallResidual { descriptor, .. } = &ops[1].kind {
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
    fn transform_graph_reports_unknowns() {
        let mut graph = FunctionGraph::new("demo");
        graph.push_op_var(
            graph.startblock,
            OpKind::Abort {
                kind: crate::model::UnknownKind::UnsupportedExpr {
                    variant: crate::model::UnsupportedExprKind::OtherExpr,
                },
            },
            false,
        );
        graph.set_raise(graph.startblock, "not implemented");
        let result = transform_graph(&graph, &GraphTransformConfig::default());
        assert_eq!(result.notes.len(), 2); // unknown + abort
    }

    #[test]
    fn transform_graph_consumes_identity_virtualizable_hints() {
        let mut graph = FunctionGraph::new("demo");
        let frame_var = graph.alloc_value_var();
        let hinted_var = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, frame_var.clone());
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["hint_access_directly"]),
                args: vec![frame_var],
                result_ty: ValueType::Ref(None),
            },
            false,
        );
        graph
            .block_mut(graph.startblock)
            .operations
            .last_mut()
            .unwrap()
            .result = Some(hinted_var.clone());
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldRead {
                base: hinted_var,
                field: crate::model::FieldDescriptor::new("next_instr", Some("Frame".into())),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        let result = transform_graph(
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
        assert_eq!(ops.len(), 2);
        assert!(
            matches!(ops[0].kind, OpKind::Live),
            "virtualizable read must be led by -live-, got {:?}",
            ops[0].kind
        );
        match &ops[1].kind {
            OpKind::VableFieldRead { field_index, .. } => assert_eq!(*field_index, 0),
            other => panic!("expected VableFieldRead after hint suppression, got {other:?}"),
        }
    }

    #[test]
    fn transform_graph_rewrites_hint_force_virtualizable() {
        let mut graph = FunctionGraph::new("demo");
        let frame_var = graph.alloc_value_var();
        let forced_var = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, frame_var.clone());
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["hint_force_virtualizable"]),
                args: vec![frame_var.clone()],
                result_ty: ValueType::Ref(None),
            },
            false,
        );
        graph
            .block_mut(graph.startblock)
            .operations
            .last_mut()
            .unwrap()
            .result = Some(forced_var.clone());
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldRead {
                base: forced_var,
                field: crate::model::FieldDescriptor::new("next_instr", Some("Frame".into())),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        let result = transform_graph(
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
        let OpKind::VableForce { base } = &ops[0].kind else {
            panic!("expected ops[0] to be VableForce, got {:?}", ops[0].kind);
        };
        assert_eq!(base, &frame_var);
        assert!(
            matches!(ops[1].kind, OpKind::Live),
            "virtualizable read must be led by -live-, got {:?}",
            ops[1].kind
        );
        assert!(matches!(
            ops[2].kind,
            OpKind::VableFieldRead { field_index: 0, .. }
        ));
    }

    /// `jtransform.py:384-392` — `rewrite_op_int_add_ovf` (aliased to
    /// `rewrite_op_int_mul_ovf`) and `rewrite_op_int_sub_ovf` both return
    /// `[SpaceOperation('-live-', [], None), op]`.  `flatten` pops the
    /// overflow-checked op back off the emitted line and re-spells it as
    /// `int_*_jump_if_ovf`, a guard that resumes at this point, so the
    /// marker must be established here rather than inherited from whatever
    /// operation happens to precede it.
    #[test]
    fn transform_graph_leads_overflow_checked_binops_with_live() {
        for opname in ["add_ovf", "sub_ovf", "mul_ovf"] {
            let mut graph = FunctionGraph::new("ovf");
            let entry = graph.startblock;
            let lhs = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
            let rhs = graph.push_op_var(entry, OpKind::ConstInt(2), true).unwrap();
            graph.push_op_var(
                entry,
                OpKind::BinOp {
                    op: opname.into(),
                    lhs,
                    rhs,
                    result_ty: crate::model::ValueType::Int,
                },
                true,
            );

            let result = transform_graph(&graph, &GraphTransformConfig::default());
            let ops = &result.graph.block(entry).operations;
            let ovf_idx = ops
                .iter()
                .position(|o| matches!(&o.kind, OpKind::BinOp { op, .. } if op == opname))
                .unwrap_or_else(|| panic!("{opname} did not survive the rewrite: {ops:?}"));
            assert!(
                ovf_idx > 0 && matches!(ops[ovf_idx - 1].kind, OpKind::Live),
                "{opname} must be led by -live-, got {ops:?}"
            );
        }
    }

    /// `jtransform.py:1748-1755` — `jit.not_in_trace`'s registered spec is
    /// `"jit.not_in_trace()"` (`rlib/jit.py:260`), an empty argtuple, so the
    /// decoded arg list comes back empty.  Upstream ignores it ("use the
    /// original 'op.args'") and the residual call keeps the arguments the
    /// call site passes.
    #[test]
    fn not_in_trace_residual_call_keeps_the_original_arguments() {
        let path = crate::parse::CallPath::from_segments(["helper", "trace_hook"]);
        let target = CallTarget::function_path(["helper", "trace_hook"]);

        let mut cc = crate::call::CallControl::new();
        cc.mark_oopspec(path.clone(), "jit.not_in_trace()".to_string());
        // A non-empty argname list is what routes the decode through
        // `parse_oopspec` + `normalize_opargs` rather than the positional
        // pass-through fallback.
        cc.mark_oopspec_argnames(path, vec!["x".into(), "y".into()]);

        let mut graph = FunctionGraph::new("caller");
        let x = graph.alloc_value_var();
        let y = graph.alloc_value_var();
        FunctionGraph::set_concretetype_of_inline(&x, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&y, ConcreteType::Signed);
        let startblock = graph.startblock;
        graph.push_op_var(
            startblock,
            OpKind::Call {
                target,
                args: vec![x.clone(), y.clone()],
                result_ty: ValueType::Void,
            },
            false,
        );

        let config = GraphTransformConfig::default();
        let transformed = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .transform(&graph);

        let args_i = transformed
            .graph
            .block(startblock)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::CallResidual { args_i, .. } => Some(args_i.clone()),
                _ => None,
            })
            .expect("not_in_trace must lower to a residual call");
        assert_eq!(args_i, vec![x, y]);
    }

    /// `rpython/jit/codewriter/jtransform.py:608-614 rewrite_op_hint`
    /// `promote=True` branch: emits `[-live-, <kind>_guard_value(x),
    /// None]` where the `None` sentinel aliases the result back to the
    /// input arg.  In pyre's `RewriteResult` model the alias is applied
    /// by `optimize_block` from `self.aliases.insert(result, base)` and
    /// the two emitted ops show up at the call site as
    /// `[OpKind::Live, OpKind::GuardValue { kind_char }]`.
    #[test]
    fn transform_graph_rewrites_hint_promote() {
        let mut graph = FunctionGraph::new("demo");
        let v_var = graph.alloc_value_var();
        let promoted_var = graph.alloc_value_var();
        let consumed_var = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, v_var.clone());
        // `hint_promote(v)` — mirrors `rlib/jit.py:101 promote(x)` after
        // lowering to the operator-level helper name.
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["hint_promote"]),
                args: vec![v_var.clone()],
                result_ty: ValueType::Ref(None),
            },
            false,
        );
        graph
            .block_mut(graph.startblock)
            .operations
            .last_mut()
            .unwrap()
            .result = Some(promoted_var.clone());
        // A downstream op that names the promote result so we can
        // observe that `optimize_block` aliased it back to `v`.
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldRead {
                base: promoted_var,
                field: crate::model::FieldDescriptor::new("payload", Some("Box".into())),
                ty: ValueType::Int,
                pure: false,
            },
            false,
        );
        graph
            .block_mut(graph.startblock)
            .operations
            .last_mut()
            .unwrap()
            .result = Some(consumed_var);
        graph.set_return(graph.startblock, None);

        let result = transform_graph(&graph, &GraphTransformConfig::default());
        let ops = &result.graph.block(graph.startblock).operations;
        // Expected post-rewrite shape: [Live, GuardValue, FieldRead].
        assert_eq!(ops.len(), 3, "got {ops:?}");
        assert!(matches!(ops[0].kind, OpKind::Live));
        match &ops[1].kind {
            OpKind::GuardValue {
                value, kind_char, ..
            } => {
                assert_eq!(value, &v_var, "guard target must remain the input arg");
                assert_eq!(*kind_char, 'r', "default kind without type-state");
            }
            other => panic!("expected GuardValue, got {other:?}"),
        }
        // `None` sentinel parity: the downstream FieldRead, which named
        // the `promoted` result, must have its base resolved back to `v`.
        match &ops[2].kind {
            OpKind::FieldRead { base, .. } => {
                assert_eq!(base, &v_var, "promote result must alias back to input arg");
            }
            other => panic!("expected FieldRead, got {other:?}"),
        }
    }

    /// `jtransform.py:608` voidness guard — `if hints.get('promote')
    /// and op.args[0].concretetype is not lltype.Void`.  Pyre falls
    /// through (Keep) when `value_kind(arg) == 'v'`.
    #[test]
    fn transform_graph_keeps_hint_promote_on_void_arg() {
        let mut graph = FunctionGraph::new("demo");
        let v_var = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, v_var.clone());
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["hint_promote"]),
                args: vec![v_var.clone()],
                result_ty: ValueType::Void,
            },
            false,
        );
        graph.set_return(graph.startblock, None);

        let config = GraphTransformConfig::default();
        // Mark `v` as void-kind on its backing Variable before
        // rewriting (mirrors RPython's `v.concretetype = lltype.Void`).
        FunctionGraph::set_concretetype_of_inline(
            &v_var,
            crate::codewriter::type_state::ConcreteType::Void,
        );
        let mut transformer = Transformer::new(&config);
        // Direct call to rewrite_operation — without setting up the
        // optimize_block plumbing — verifies the Keep result for the
        // void-kind branch in isolation.
        let op = graph
            .block(graph.startblock)
            .operations
            .last()
            .unwrap()
            .clone();
        assert!(matches!(
            transformer.rewrite_operation(&op, "demo", &mut graph),
            RewriteResult::Keep
        ));
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
        let base_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "cell".to_string(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldRead {
                base: base_var,
                field: FieldDescriptor::new("value", Some("Cell".to_string())),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

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
        let base_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "p".to_string(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldRead {
                base: base_var,
                field: FieldDescriptor::new("x", Some("Point".to_string())),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

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

    #[test]
    fn handle_jit_marker_loop_header_emits_single_loop_header_op() {
        // jtransform.py:1714-1718 `SpaceOperation('loop_header', [c_index], None)`.
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config);
        let ops = transformer.handle_jit_marker__loop_header(7);
        assert_eq!(ops.len(), 1);
        match &ops[0].kind {
            OpKind::LoopHeader { jitdriver_index } => assert_eq!(*jitdriver_index, 7),
            other => panic!("expected OpKind::LoopHeader, got {other:?}"),
        }
        assert!(ops[0].result.is_none(), "loop_header produces no result");
    }

    #[test]
    fn handle_jit_marker_jit_merge_point_emits_live_merge_live_sequence() {
        // jtransform.py:1707-1712 — return shape is `ops + [op3, op1, op2]`
        // where op3=live_preamble, op1=jit_merge_point, op2=live_recursive.
        use crate::flowspace::model::Variable;
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_portal_jd(Some(3));
        let green_i = Variable::new();
        let red_i_a = Variable::new();
        let red_i_b = Variable::new();
        let red_r = Variable::new();
        let ops = transformer.handle_jit_marker__jit_merge_point(
            vec![green_i.clone()],
            vec![],
            vec![],
            vec![red_i_a.clone(), red_i_b.clone()],
            vec![red_r.clone()],
            vec![],
        );
        assert_eq!(ops.len(), 3, "expect live + merge + live");
        assert!(matches!(ops[0].kind, OpKind::Live));
        match &ops[1].kind {
            OpKind::JitMergePoint {
                jitdriver_index,
                greens_i,
                reds_i,
                reds_r,
                ..
            } => {
                assert_eq!(*jitdriver_index, 3);
                assert_eq!(greens_i, &vec![green_i]);
                assert_eq!(reds_i, &vec![red_i_a, red_i_b]);
                assert_eq!(reds_r, &vec![red_r]);
            }
            other => panic!("expected OpKind::JitMergePoint, got {other:?}"),
        }
        assert!(matches!(ops[2].kind, OpKind::Live));
    }

    #[test]
    fn autodetect_jit_markers_redvars_finds_live_non_green_vars_in_kind_order() {
        use crate::codewriter::type_state::ConcreteType;
        use crate::flowspace::model::Variable;
        use crate::model::{Link, LinkArg, ValueType};

        let mut graph = crate::model::FunctionGraph::new("autoreds_portal");
        let receiver = Variable::named("driver");
        let green = Variable::named("greenkey");
        let remapped_green = Variable::named("greenkey_alias");
        let red_i = Variable::named("red_i");
        let w_iterator = Variable::named("w_iterator");
        let items = Variable::named("items");
        let red_f = Variable::named("red_f");
        let void = Variable::named("void");
        let last_exception = Variable::named("last_exception");
        let last_exc_value = Variable::named("last_exc_value");
        let constant = Variable::named("constant");
        let after_result = Variable::named("after_result");
        for variable in [&receiver, &green, &remapped_green, &w_iterator, &items] {
            FunctionGraph::set_concretetype_of_inline(variable, ConcreteType::GcRef);
        }
        FunctionGraph::set_concretetype_of_inline(&red_i, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&red_f, ConcreteType::Float);
        FunctionGraph::set_concretetype_of_inline(&void, ConcreteType::Void);
        FunctionGraph::set_concretetype_of_inline(&constant, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&after_result, ConcreteType::Signed);

        graph.blocks[graph.startblock.0]
            .operations
            .push(SpaceOperation {
                result: Some(constant.clone()),
                kind: OpKind::ConstInt(7),
            });
        graph.blocks[graph.startblock.0]
            .operations
            .push(SpaceOperation {
                result: None,
                kind: OpKind::Call {
                    target: CallTarget::method(
                        "jit_merge_point",
                        Some("UnpackIterableJitDriver".into()),
                    ),
                    args: vec![receiver.clone(), green.clone()],
                    result_ty: ValueType::Void,
                },
            });
        // Reverse liveness must discard this result and add its operands.
        graph.blocks[graph.startblock.0]
            .operations
            .push(SpaceOperation {
                result: Some(after_result.clone()),
                kind: OpKind::BinOp {
                    op: "int_add".into(),
                    lhs: red_i.clone(),
                    rhs: green.clone(),
                    result_ty: ValueType::Int,
                },
            });
        let link_args = vec![
            red_i.clone(),
            w_iterator.clone(),
            items.clone(),
            red_f.clone(),
            green.clone(),
            receiver,
            void,
            constant,
            after_result,
            last_exception.clone(),
            last_exc_value.clone(),
        ];
        let (target, _) = graph.create_block_with_arg_vars(link_args.len());
        graph.blocks[graph.startblock.0].exits = vec![
            Link::new_mixed(
                link_args.into_iter().map(LinkArg::Value).collect(),
                target,
                None,
            )
            .extravars(
                Some(LinkArg::Value(last_exception)),
                Some(LinkArg::Value(last_exc_value)),
            ),
        ];

        assert_eq!(
            autodetect_jit_markers_redvars(&graph, &[remapped_green]),
            vec![red_i, w_iterator, items, red_f],
            "reds must be INT<REF<FLOAT with deterministic Variable-id order within a kind"
        );
    }

    #[test]
    fn jit_marker_key_recognises_allow_listed_jitdriver_methods() {
        let unpack_merge =
            CallTarget::method("jit_merge_point", Some("UnpackIterableJitDriver".into()));
        assert_eq!(
            jit_marker_key_from_target(&unpack_merge),
            Some(JitMarkerKey::JitMergePoint)
        );

        let merge = CallTarget::method("jit_merge_point", Some("PyPyJitDriver".into()));
        assert_eq!(
            jit_marker_key_from_target(&merge),
            Some(JitMarkerKey::JitMergePoint)
        );
        let cej = CallTarget::method("can_enter_jit", Some("PyPyJitDriver".into()));
        assert_eq!(
            jit_marker_key_from_target(&cej),
            Some(JitMarkerKey::CanEnterJit)
        );
        let lh = CallTarget::method("loop_header", Some("PyPyJitDriver".into()));
        assert_eq!(
            jit_marker_key_from_target(&lh),
            Some(JitMarkerKey::LoopHeader)
        );
        // Other receivers or other methods must not match.
        let other = CallTarget::method("jit_merge_point", Some("SomeOtherType".into()));
        assert_eq!(jit_marker_key_from_target(&other), None);
        let missing_receiver = CallTarget::method("jit_merge_point", None);
        assert_eq!(jit_marker_key_from_target(&missing_receiver), None);
        let other_method = CallTarget::method("not_a_marker", Some("PyPyJitDriver".into()));
        assert_eq!(jit_marker_key_from_target(&other_method), None);
        // Non-method targets are never markers.
        let free_fn = CallTarget::function_path(["module", "jit_merge_point"]);
        assert_eq!(jit_marker_key_from_target(&free_fn), None);
    }

    #[test]
    fn source_driver_calls_lower_to_jitcode_markers_not_residual_calls() {
        use crate::codewriter::call::CallControl;
        use crate::codewriter::type_state::ConcreteType;
        use crate::parse::CallPath;

        let mut cc = CallControl::new();
        cc.setup_jitdriver(
            CallPath::from_segments(["pyre_jit", "other_portal"]),
            Vec::new(),
            Vec::new(),
            false,
            Vec::new(),
            Vec::new(),
        );
        let portal_graph = CallPath::from_segments(["pyre_jit", "eval_loop_jit"]);
        cc.setup_jitdriver(
            portal_graph.clone(),
            vec![
                "next_instr".into(),
                "is_being_profiled".into(),
                "pycode".into(),
            ],
            vec!["frame".into(), "ec".into()],
            false,
            Vec::new(),
            Vec::new(),
        );
        let resolved_driver = cc
            .jitdriver_sd_from_portal_graph(&portal_graph)
            .expect("portal graph must resolve its own driver")
            .index;
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .with_portal_jd(Some(resolved_driver));
        let mut graph = crate::model::FunctionGraph::new("eval_loop_jit_source_markers");
        let receiver = graph.alloc_value_var();
        let next_instr = graph.alloc_value_var();
        let profiled = graph.alloc_value_var();
        let pycode = graph.alloc_value_var();
        let frame = graph.alloc_value_var();
        let ec = graph.alloc_value_var();
        FunctionGraph::set_concretetype_of_inline(&next_instr, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&profiled, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&pycode, ConcreteType::GcRef);
        FunctionGraph::set_concretetype_of_inline(&frame, ConcreteType::GcRef);
        FunctionGraph::set_concretetype_of_inline(&ec, ConcreteType::GcRef);
        let merge_target = CallTarget::method("jit_merge_point", Some("PyPyJitDriver".into()));
        let merge_key = jit_marker_key_from_target(&merge_target)
            .expect("source jit_merge_point method must be recognized");
        let merge_ops = transformer
            .try_handle_jit_marker(
                merge_key,
                &[receiver, next_instr, profiled, pycode, frame, ec],
                &graph,
            )
            .expect("recognized portal marker must lower");
        assert!(
            merge_ops
                .iter()
                .any(|op| matches!(op.kind, OpKind::JitMergePoint { .. }))
        );
        assert!(
            !merge_ops
                .iter()
                .any(|op| matches!(op.kind, OpKind::Call { .. })),
            "jit_merge_point must not remain a residual call"
        );

        let enter_target = CallTarget::method("can_enter_jit", Some("PyPyJitDriver".into()));
        let enter_key = jit_marker_key_from_target(&enter_target)
            .expect("source can_enter_jit method must be recognized");
        let enter_ops = transformer
            .try_handle_jit_marker(enter_key, &[], &graph)
            .expect("recognized can_enter_jit marker must lower");
        assert!(matches!(
            enter_ops.as_slice(),
            [SpaceOperation {
                kind: OpKind::LoopHeader { jitdriver_index },
                ..
            }] if *jitdriver_index == resolved_driver
        ));
        assert!(
            !enter_ops
                .iter()
                .any(|op| matches!(op.kind, OpKind::Call { .. })),
            "can_enter_jit must not remain a residual call"
        );
    }

    #[test]
    fn try_handle_jit_marker_can_enter_jit_aliases_to_loop_header() {
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_portal_jd(Some(2));
        let ops = transformer
            .try_handle_jit_marker(
                JitMarkerKey::CanEnterJit,
                &[],
                &crate::model::FunctionGraph::new("fixture"),
            )
            .expect("can_enter_jit should dispatch when portal_jd is set");
        assert_eq!(ops.len(), 1);
        match &ops[0].kind {
            OpKind::LoopHeader { jitdriver_index } => assert_eq!(*jitdriver_index, 2),
            other => panic!("expected LoopHeader, got {other:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "reds='auto' is supported only for jit drivers")]
    fn try_handle_jit_marker_rejects_can_enter_jit_on_an_autoreds_driver() {
        // `support.py:77-80` — the auto-detected red set comes from the
        // jit_merge_point's own block, so a second marker kind on the same
        // driver has no consistent red set and upstream refuses to translate.
        let config = GraphTransformConfig::default();
        let mut cc = crate::call::CallControl::new();
        cc.setup_jitdriver(
            crate::parse::CallPath::from_segments(["autoreds_portal"]),
            vec!["greenkey".to_string()],
            Vec::new(),
            true,
            Vec::new(),
            Vec::new(),
        );
        let mut transformer = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .with_portal_jd(Some(0));
        let _ = transformer.try_handle_jit_marker(
            JitMarkerKey::CanEnterJit,
            &[],
            &crate::model::FunctionGraph::new("fixture"),
        );
    }

    #[test]
    fn try_handle_jit_marker_drops_markers_for_inactive_driver() {
        // jtransform.py:1661-1662 `if not jitdriver.active: return []`. A
        // deactivated portal driver dispatches but drops its markers; an
        // active driver lowers them normally.
        let config = GraphTransformConfig::default();
        let mut cc = crate::call::CallControl::new();
        cc.setup_jitdriver(
            crate::parse::CallPath::from_segments(["portal"]),
            Vec::new(),
            Vec::new(),
            false,
            Vec::new(),
            Vec::new(),
        );

        cc.set_jitdriver_active(0, false);
        {
            let mut transformer = Transformer::new(&config)
                .with_callcontrol(&mut cc)
                .with_portal_jd(Some(0));
            let ops = transformer
                .try_handle_jit_marker(
                    JitMarkerKey::LoopHeader,
                    &[],
                    &crate::model::FunctionGraph::new("fixture"),
                )
                .expect("inactive driver still dispatches (then drops)");
            assert!(
                ops.is_empty(),
                "inactive driver must drop markers, got {ops:?}"
            );
        }

        cc.set_jitdriver_active(0, true);
        let mut transformer = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .with_portal_jd(Some(0));
        let ops = transformer
            .try_handle_jit_marker(
                JitMarkerKey::LoopHeader,
                &[],
                &crate::model::FunctionGraph::new("fixture"),
            )
            .expect("active driver dispatches");
        assert_eq!(ops.len(), 1);
        assert!(
            matches!(ops[0].kind, OpKind::LoopHeader { jitdriver_index: 0 }),
            "active driver lowers LoopHeader, got {:?}",
            ops[0].kind
        );
    }

    #[test]
    fn promote_greens_emits_live_guard_value_pair_per_green() {
        // jtransform.py:1646-1656. One `-live-` + `{kind}_guard_value` pair
        // per green, in input order. Without a type_state every green falls
        // back to kind 'r'.
        let config = GraphTransformConfig::default();
        let transformer = Transformer::new(&config);
        let mut graph = crate::model::FunctionGraph::new("test_promote_greens_fixture");
        let greens: Vec<crate::flowspace::model::Variable> =
            (0..3).map(|_| graph.alloc_value_var()).collect();
        let ops = transformer.promote_greens(&greens);
        assert_eq!(ops.len(), 6, "expect 2 ops per green");
        for i in 0..greens.len() {
            assert!(
                matches!(ops[i * 2].kind, OpKind::Live),
                "slot {i} should start with Live"
            );
            match &ops[i * 2 + 1].kind {
                OpKind::GuardValue {
                    value, kind_char, ..
                } => {
                    assert_eq!(value, &greens[i]);
                    assert_eq!(*kind_char, 'r');
                }
                other => panic!("slot {i} expected GuardValue, got {other:?}"),
            }
        }
    }

    #[test]
    fn promote_greens_empty_input_yields_empty_output() {
        let config = GraphTransformConfig::default();
        let transformer = Transformer::new(&config);
        assert!(transformer.promote_greens(&[]).is_empty());
    }

    #[test]
    fn try_handle_jit_marker_returns_none_without_portal() {
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config);
        // No portal_jd set → dispatch is a no-op (caller falls through).
        assert!(
            transformer
                .try_handle_jit_marker(
                    JitMarkerKey::LoopHeader,
                    &[],
                    &crate::model::FunctionGraph::new("fixture")
                )
                .is_none()
        );
    }

    #[test]
    #[should_panic(expected = "'jit_merge_point' in non-portal graph!")]
    fn handle_jit_marker_jit_merge_point_without_portal_panics() {
        // jtransform.py:1691 `assert self.portal_jd is not None`.
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config);
        transformer.handle_jit_marker__jit_merge_point(
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
        );
    }

    #[test]
    #[should_panic(expected = "duplicate red variable on jit_merge_point()")]
    fn handle_jit_marker_jit_merge_point_rejects_duplicate_red() {
        // jtransform.py:1702 `assert len(dict.fromkeys(redlist)) ==
        // len(list(redlist))` — a red Variable repeated within its
        // kind-list is rejected. `Variable::clone` aliases the same
        // identity, so passing one Variable twice is an exact duplicate.
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_portal_jd(Some(0));
        let mut graph = crate::model::FunctionGraph::new("dup_red_fixture");
        let r = graph.alloc_value_var();
        transformer.handle_jit_marker__jit_merge_point(
            vec![],
            vec![],
            vec![],
            vec![r.clone(), r],
            vec![],
            vec![],
        );
    }

    /// Upstream parity test for the entry-level dispatch shape — a
    /// straight port of `test_jtransform.py:1011-1046 test_jit_merge_point_1`.
    ///
    /// Input: `try_handle_jit_marker(JitMergePoint, [receiver, g1, g2,
    /// r1])` with two greens and one red, all int-typed via direct
    /// `FunctionGraph::set_concretetype_of_inline`. Expected output (`promote_greens` +
    /// `handle_jit_marker__jit_merge_point`):
    ///
    /// ```text
    /// op0 = -live-
    /// op1 = int_guard_value(g1)
    /// op2 = -live-
    /// op3 = int_guard_value(g2)
    /// op4 = -live-                                   ← live_preamble
    /// op5 = jit_merge_point(idx, I[g1,g2], R[], F[], ← merge
    ///                       I[r1],     R[], F[])
    /// op6 = -live-                                   ← live_recursive
    /// ```
    ///
    /// Locks the entry-level `try_handle_jit_marker` ↔
    /// `promote_greens` ↔ `handle_jit_marker__jit_merge_point`
    /// composition shape so a regression that drops the promote_greens
    /// prefix or the trailing live ops fails immediately.
    #[test]
    fn try_handle_jit_marker_jit_merge_point_emits_full_promote_greens_sequence() {
        use crate::codewriter::call::CallControl;
        use crate::codewriter::type_state::ConcreteType;
        use crate::parse::CallPath;

        let mut cc = CallControl::new();
        cc.setup_jitdriver(
            CallPath::from_segments(["test", "portal"]),
            vec!["green1".into(), "green2".into()],
            vec!["red1".into()],
            false,
            Vec::new(),
            Vec::new(),
        );

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .with_portal_jd(Some(0));

        // promote_greens reads Variable.concretetype directly to populate
        // the GuardValue.value field. Mint four distinct values; jtransform
        // reads operand kinds via Variable.concretetype, so hydrate the two
        // greens and the red below to `'i'` instead of the Unknown-defaulted
        // `'r'`.
        let mut graph = crate::model::FunctionGraph::new("test_jit_merge_point_fixture");
        let receiver_var = graph.alloc_value_var();
        let g1_var = graph.alloc_value_var();
        let g2_var = graph.alloc_value_var();
        let r1_var = graph.alloc_value_var();
        FunctionGraph::set_concretetype_of_inline(&g1_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&g2_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&r1_var, ConcreteType::Signed);
        let args_vars = vec![receiver_var, g1_var.clone(), g2_var.clone(), r1_var.clone()];
        let ops = transformer
            .try_handle_jit_marker(JitMarkerKey::JitMergePoint, &args_vars, &graph)
            .expect("portal_jd + cc + 2-greens + 1-red satisfies dispatch preconditions");

        assert_eq!(ops.len(), 7, "promote_greens(2 greens)*2 + merge*3 = 7");

        // promote_greens prefix: -live-, int_guard_value(g1), -live-, int_guard_value(g2)
        assert!(matches!(ops[0].kind, OpKind::Live));
        match &ops[1].kind {
            OpKind::GuardValue {
                value, kind_char, ..
            } => {
                assert_eq!(value, &g1_var);
                assert_eq!(*kind_char, 'i');
            }
            other => panic!("ops[1] expected GuardValue(g1, 'i'), got {other:?}"),
        }
        assert!(matches!(ops[2].kind, OpKind::Live));
        match &ops[3].kind {
            OpKind::GuardValue {
                value, kind_char, ..
            } => {
                assert_eq!(value, &g2_var);
                assert_eq!(*kind_char, 'i');
            }
            other => panic!("ops[3] expected GuardValue(g2, 'i'), got {other:?}"),
        }
        // live_preamble + merge + live_recursive
        assert!(matches!(ops[4].kind, OpKind::Live));
        match &ops[5].kind {
            OpKind::JitMergePoint {
                jitdriver_index,
                greens_i,
                greens_r,
                greens_f,
                reds_i,
                reds_r,
                reds_f,
            } => {
                assert_eq!(*jitdriver_index, 0);
                assert_eq!(greens_i, &vec![g1_var.clone(), g2_var.clone()]);
                assert!(greens_r.is_empty());
                assert!(greens_f.is_empty());
                assert_eq!(reds_i, &vec![r1_var.clone()]);
                assert!(reds_r.is_empty());
                assert!(reds_f.is_empty());
            }
            other => panic!("ops[5] expected JitMergePoint, got {other:?}"),
        }
        assert!(matches!(ops[6].kind, OpKind::Live));
        assert!(
            ops[5].result.is_none(),
            "jit_merge_point produces no result"
        );
    }

    /// jtransform.py:1699-1701 — `assert isinstance(v, Variable),
    /// "Constant specified red in jit_merge_point()"`.  front::mir
    /// materialises a Constant red into a Const-defined Variable, so
    /// the provenance scan must reject a red whose defining op is
    /// `ConstInt`.
    #[test]
    #[should_panic(expected = "Constant specified red in jit_merge_point()")]
    fn try_handle_jit_marker_rejects_constant_red() {
        use crate::codewriter::call::CallControl;
        use crate::codewriter::type_state::ConcreteType;
        use crate::parse::CallPath;

        let mut cc = CallControl::new();
        cc.setup_jitdriver(
            CallPath::from_segments(["test", "portal"]),
            vec!["green1".into()],
            vec!["red1".into()],
            false,
            Vec::new(),
            Vec::new(),
        );

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .with_portal_jd(Some(0));

        let mut graph = crate::model::FunctionGraph::new("constant_red_fixture");
        let receiver_var = graph.alloc_value_var();
        let g1_var = graph.alloc_value_var();
        let red_var = graph.alloc_value_var();
        FunctionGraph::set_concretetype_of_inline(&g1_var, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&red_var, ConcreteType::Signed);
        // The red is the image of a source-level Constant: its
        // defining op in the graph is `ConstInt`.
        graph.blocks[0].operations.push(SpaceOperation {
            result: Some(red_var.clone()),
            kind: OpKind::ConstInt(7),
        });
        let args_vars = vec![receiver_var, g1_var, red_var];
        let _ = transformer.try_handle_jit_marker(JitMarkerKey::JitMergePoint, &args_vars, &graph);
    }

    /// Synthetic Rust-wrapper elision must accept a single-arg `Ok(_)`
    /// whose result type matches the arg type and whose frontend target was
    /// resolved as a synthetic transparent constructor.
    #[test]
    fn synthetic_result_ctor_identity_accepts_prelude_ok() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("synth_ctor_ok");
        let arg = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let transformer = Transformer::new(&config);
        assert!(transformer.is_synthetic_result_option_ctor(
            &CallTarget::synthetic_transparent_ctor("Ok"),
            &[arg],
            &ValueType::Ref(None),
        ));
    }

    /// Reject a normal function call named `Ok`; user-function protection now
    /// lives in the frontend resolver, so jtransform only trusts the explicit
    /// synthetic target variant.
    #[test]
    fn synthetic_result_ctor_identity_rejects_function_path_ok() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("synth_ctor_fn_path");
        let arg = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let transformer = Transformer::new(&config);
        assert!(!transformer.is_synthetic_result_option_ctor(
            &CallTarget::function_path(["Ok"]),
            &[arg],
            &ValueType::Ref(None),
        ));
    }

    /// Reject when the target is not the explicit synthetic variant even if
    /// the spelling looks like a prelude constructor.
    #[test]
    fn synthetic_result_ctor_identity_rejects_name_only_matching() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("synth_ctor_name_only");
        let arg = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let transformer = Transformer::new(&config);
        assert!(!transformer.is_synthetic_result_option_ctor(
            &CallTarget::function_path(["Ok"]),
            &[arg],
            &ValueType::Ref(None),
        ));
    }

    /// `__cast_pointer/<Root>` marker folds to the operand alias —
    /// `rewrite_op_cast_pointer` → `rewrite_op_same_as`
    /// (jtransform.py:254-257) emits no jitcode op.
    #[test]
    fn cast_pointer_marker_elides_to_operand_alias() {
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config);
        let mut graph = FunctionGraph::new("cast_ptr_marker");
        let arg = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let result_var = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let target = CallTarget::function_path(["__cast_pointer", "W_CastTarget"]);
        let result_ty = ValueType::Ref(Some("W_CastTarget".into()));
        let op = SpaceOperation {
            result: Some(result_var),
            kind: OpKind::Call {
                target: target.clone(),
                args: vec![arg.clone()],
                result_ty: result_ty.clone(),
            },
        };
        let rewritten = transformer.rewrite_op_direct_call(
            &op,
            &target,
            std::slice::from_ref(&arg),
            &result_ty,
            "cast_ptr_marker",
            &mut graph,
        );
        match rewritten {
            RewriteResult::Identity(alias) => assert_eq!(alias, arg),
            _ => panic!("expected Identity alias to the operand"),
        }
    }

    /// `rpython/rtyper/rbuiltin.py:222-225`: `rtype_intmask` returns its
    /// single Signed input unchanged.
    #[test]
    fn rarithmetic_intmask_elides_only_the_exact_single_arg_call() {
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config);
        let mut graph = FunctionGraph::new("rarithmetic_intmask");
        let arg = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let extra = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let result_var = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let target = CallTarget::function_path(["rpython", "rlib", "rarithmetic", "intmask"]);
        let result_ty = ValueType::Int;
        let op = SpaceOperation {
            result: Some(result_var),
            kind: OpKind::Call {
                target: target.clone(),
                args: vec![arg.clone()],
                result_ty: result_ty.clone(),
            },
        };

        let rewritten = transformer.rewrite_op_direct_call(
            &op,
            &target,
            std::slice::from_ref(&arg),
            &result_ty,
            "rarithmetic_intmask",
            &mut graph,
        );
        assert!(matches!(rewritten, RewriteResult::Identity(alias) if alias == arg));

        let wrong_target = CallTarget::function_path(["rlib", "rarithmetic", "intmask"]);
        assert!(!matches!(
            transformer.rewrite_op_direct_call(
                &op,
                &wrong_target,
                std::slice::from_ref(&arg),
                &result_ty,
                "rarithmetic_intmask_wrong_target",
                &mut graph,
            ),
            RewriteResult::Identity(_)
        ));
        assert!(!matches!(
            transformer.rewrite_op_direct_call(
                &op,
                &target,
                &[arg, extra],
                &result_ty,
                "rarithmetic_intmask_wrong_arity",
                &mut graph,
            ),
            RewriteResult::Identity(_)
        ));
    }

    /// `jtransform.py:116-118` + `:176-179
    /// _killed_exception_raising_operation`: the exception exits belong to
    /// the block's last operation, so once that operation rewrites to
    /// nothing they have to go too.
    #[test]
    fn elided_last_op_drops_the_blocks_exception_exits() {
        use crate::model::{ExitSwitch, Link, exception_exitcase};

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config);
        let mut graph = FunctionGraph::new("killed_exc_op");
        let entry = graph.startblock;

        // `cast_int_to_uint` is one of the no-op casts that rewrite to
        // `RewriteResult::Identity`, i.e. contribute no operation at all.
        let operand = graph.push_op_var(entry, OpKind::ConstInt(7), true).unwrap();
        let cast = graph
            .push_op_var(
                entry,
                OpKind::UnaryOp {
                    op: "cast_int_to_uint".to_string(),
                    operand,
                    result_ty: ValueType::Unsigned,
                },
                true,
            )
            .unwrap();

        let continuation = graph.create_block();
        let phi = graph.alloc_value_var();
        graph.push_inputarg_var(continuation, phi.clone());
        graph.set_return(continuation, Some(phi));

        let (exc_block, last_exception_var, last_exc_value_var) = graph.exceptblock_arg_vars();
        let normal_link = Link::from_variables(&graph, vec![cast], continuation, None);
        let exc_link = Link::from_variables(
            &graph,
            vec![last_exception_var.clone(), last_exc_value_var.clone()],
            exc_block,
            Some(exception_exitcase()),
        )
        .extravars(
            Some(LinkArg::Value(last_exception_var)),
            Some(LinkArg::Value(last_exc_value_var)),
        );
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::LastException),
            vec![normal_link, exc_link],
        );
        assert!(graph.blocks[entry.0].canraise());

        transformer.optimize_block(&mut graph, entry.0, "killed_exc_op", exc_block);

        let block = &graph.blocks[entry.0];
        assert!(
            block.exitswitch.is_none(),
            "the last_exception exitswitch must go with the elided operation"
        );
        assert_eq!(
            block.exits.len(),
            1,
            "the exception exit must go with the elided operation"
        );
        assert_eq!(
            block.exits[0].target, continuation,
            "the surviving exit must be the normal continuation"
        );
        assert!(
            !block.exits.iter().any(|link| link.target == exc_block),
            "no exception edge may survive for an operation that is gone"
        );
    }

    /// A block that never had an operation has no *last* operation, so
    /// `jtransform.py:117 len(newoperations) == count_before_last_operation`
    /// does not apply to it — upstream never binds the name and the
    /// comparison cannot come out true.  The exception exits must survive.
    #[test]
    fn opless_block_keeps_its_exception_exits() {
        use crate::model::{ExitSwitch, Link, exception_exitcase};

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config);
        let mut graph = FunctionGraph::new("opless_exc_block");
        let entry = graph.startblock;

        let continuation = graph.create_block();
        graph.set_return(continuation, None);

        let (exc_block, last_exception_var, last_exc_value_var) = graph.exceptblock_arg_vars();
        let normal_link = Link::from_variables(&graph, vec![], continuation, None);
        let exc_link = Link::from_variables(
            &graph,
            vec![last_exception_var.clone(), last_exc_value_var.clone()],
            exc_block,
            Some(exception_exitcase()),
        )
        .extravars(
            Some(LinkArg::Value(last_exception_var)),
            Some(LinkArg::Value(last_exc_value_var)),
        );
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::LastException),
            vec![normal_link, exc_link],
        );
        assert!(graph.blocks[entry.0].operations.is_empty());
        assert!(graph.blocks[entry.0].canraise());

        transformer.optimize_block(&mut graph, entry.0, "opless_exc_block", exc_block);

        let block = &graph.blocks[entry.0];
        assert_eq!(
            block.exitswitch,
            Some(ExitSwitch::LastException),
            "a block with no operations has no elided last operation"
        );
        assert_eq!(
            block.exits.len(),
            2,
            "the exception exit must survive a block that never had an operation"
        );
    }

    #[test]
    fn niladic_result_variant_lowers_to_new_and_tag_store() {
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config);
        let mut graph = FunctionGraph::new("result_variant_malloc");
        let result_var = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let owner = "core::result::Result<i64,PyError>::Err".to_string();
        let target = CallTarget::synthetic_transparent_ctor_with_owner(
            vec![
                "core".to_string(),
                "result".to_string(),
                "Result<i64,PyError>".to_string(),
            ],
            "Err",
        );
        let result_ty = ValueType::Ref(Some(owner.clone()));
        let op = SpaceOperation {
            result: Some(result_var.clone()),
            kind: OpKind::Call {
                target: target.clone(),
                args: vec![],
                result_ty: result_ty.clone(),
            },
        };

        let RewriteResult::Replace(ops) = transformer.rewrite_op_direct_call(
            &op,
            &target,
            &[],
            &result_ty,
            "result_variant_malloc",
            &mut graph,
        ) else {
            panic!("niladic Result variant must lower to malloc + tag store");
        };
        assert_eq!(ops.len(), 2);
        assert!(matches!(
            &ops[0],
            SpaceOperation {
                result: Some(result),
                kind: OpKind::New { owner: allocated },
            } if result == &result_var && allocated == &owner
        ));
        assert!(matches!(
            &ops[1].kind,
            OpKind::FieldWrite {
                base,
                field,
                value: crate::model::LinkArg::Const(value),
                ty: ValueType::Int,
            } if base == &result_var
                && field.name == "__discriminant"
                && field.owner_root.as_deref()
                    == Some("core::result::Result<i64,PyError>")
                && value.value == crate::flowspace::model::ConstValue::Int(1)
        ));
    }

    /// `fold_we_are_jitted_calls` rewrites the `we_are_jitted()`
    /// `direct_call` to the `_we_are_jitted` symbolic constant — the
    /// model-graph counterpart of RPython's rtyper `specialize_call`
    /// (`rpython/rlib/jit.py:403-406`).
    #[test]
    fn we_are_jitted_specializes_to_symbolic() {
        let mut graph = FunctionGraph::new("we_are_jitted_specialize");
        let entry = graph.startblock;
        let target = CallTarget::function_path(["majit_metainterp", "jit", "we_are_jitted"]);
        let result_var = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target,
                    args: vec![],
                    result_ty: ValueType::Bool,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(result_var.clone()));
        fold_we_are_jitted_calls(&mut graph);
        let op = graph.blocks[0]
            .operations
            .iter()
            .find(|op| op.result.as_ref() == Some(&result_var))
            .expect("we_are_jitted op present");
        match &op.kind {
            OpKind::ConstSymbolic { tag, .. } => assert_eq!(
                *tag,
                crate::translator::backendopt::constfold::WE_ARE_JITTED_TAG_ID
            ),
            other => panic!("expected ConstSymbolic, got {other:?}"),
        }
    }

    /// End-to-end: `we_are_jitted()` folds to `ConstBool(true)` in
    /// jitcode (`fold_we_are_jitted_calls` → symbolic →
    /// `rewrite_operation` `SpecTag`-identity fold, mirroring
    /// `rewrite_op_int_is_true` of `_we_are_jitted`,
    /// jtransform.py:1636-1639) — the jitcode runs during tracing and
    /// blackholing where the JIT-mode flag is true (rlib/jit.py:355).
    #[test]
    fn we_are_jitted_folds_to_const_true() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("we_are_jitted_fold");
        let entry = graph.startblock;
        let target = CallTarget::function_path(["majit_metainterp", "jit", "we_are_jitted"]);
        let result_var = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target,
                    args: vec![],
                    result_ty: ValueType::Bool,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(result_var.clone()));
        let result = transform_graph(&graph, &config);
        let folded = result
            .graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .find(|op| op.result.as_ref() == Some(&result_var))
            .expect("we_are_jitted op must survive as a const-define");
        assert!(matches!(folded.kind, OpKind::ConstBool(true)));
    }

    /// Reject when arg/result IR kinds disagree — that means the
    /// callee is doing real boxing work, not a transparent wrapper.
    #[test]
    fn synthetic_result_ctor_identity_rejects_kind_mismatch() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("synth_ctor_kind_mismatch");
        let arg = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let transformer = Transformer::new(&config);
        assert!(!transformer.is_synthetic_result_option_ctor(
            &CallTarget::synthetic_transparent_ctor("Ok"),
            &[arg],
            &ValueType::Ref(None),
        ));
    }

    #[test]
    fn ptr_null_builtin_rewrites_to_null_ref_constant() {
        for path in [
            vec!["core", "ptr", "null_mut"],
            vec!["pyre_object", "pyobject", "PY_NULL"],
        ] {
            let config = GraphTransformConfig::default();
            let mut graph = FunctionGraph::new("ptr_null_constant");
            let entry = graph.startblock;
            let result_var = graph
                .push_op_var(
                    entry,
                    OpKind::Call {
                        target: CallTarget::function_path(path),
                        args: vec![],
                        result_ty: ValueType::Ref(None),
                    },
                    true,
                )
                .unwrap();
            FunctionGraph::set_concretetype_of_inline(&result_var, ConcreteType::GcRef);
            graph.set_return(entry, Some(result_var.clone()));

            let result = transform_graph(&graph, &config);
            let folded = result
                .graph
                .blocks
                .iter()
                .flat_map(|block| &block.operations)
                .find(|op| op.result.as_ref() == Some(&result_var))
                .expect("null result must survive as a constant definition");
            assert!(matches!(folded.kind, OpKind::ConstRefNull));
        }
    }

    /// `rtuple.py:153-169 TupleRepr.newtuple`: a non-empty tuple lowers to
    /// `malloc(GcStruct)`; the following per-item `FieldWrite`s supply the
    /// `setfield`s.  It must never survive as a synthetic residual call.
    #[test]
    fn nonempty_synthetic_tuple_ctor_lowers_to_new() {
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config);
        let mut graph = FunctionGraph::new("nonempty_tuple_malloc");
        let result_var = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let owner = "Tuple<PyObjectRef,Option<PyObjectRef>>".to_string();
        let target = CallTarget::synthetic_transparent_ctor(owner.clone());
        let result_ty = ValueType::Ref(Some(owner.clone()));
        let op = SpaceOperation {
            result: Some(result_var.clone()),
            kind: OpKind::Call {
                target: target.clone(),
                args: vec![],
                result_ty: result_ty.clone(),
            },
        };

        let RewriteResult::Replace(ops) = transformer.rewrite_op_direct_call(
            &op,
            &target,
            &[],
            &result_ty,
            "nonempty_tuple_malloc",
            &mut graph,
        ) else {
            panic!("non-empty Tuple aggregate must lower to malloc");
        };
        assert!(matches!(
            ops.as_slice(),
            [SpaceOperation {
                result: Some(result),
                kind: OpKind::New { owner: allocated },
            }] if result == &result_var && allocated == &owner
        ));
    }

    /// The universal MIR tuple aggregate has no Rust nominal owner, so the
    /// malloc arm is gated on an EMPTY `owner_path`.  A user ADT whose leaf
    /// merely spells like a tuple shape carries an owner path and follows its
    /// own allocation policy; it must not be captured by that arm.
    #[test]
    fn owned_tuple_named_synthetic_ctor_does_not_take_the_tuple_malloc_arm() {
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config);
        let mut graph = FunctionGraph::new("owned_tuple_named_ctor");
        let result_var = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let owner = "mymod::Tuple<PyObjectRef>".to_string();
        let target = CallTarget::synthetic_transparent_ctor_with_owner(
            vec!["mymod".to_string()],
            "Tuple<PyObjectRef>",
        );
        let result_ty = ValueType::Ref(Some(owner.clone()));
        let op = SpaceOperation {
            result: Some(result_var.clone()),
            kind: OpKind::Call {
                target: target.clone(),
                args: vec![],
                result_ty: result_ty.clone(),
            },
        };

        let rewritten = transformer.rewrite_op_direct_call(
            &op,
            &target,
            &[],
            &result_ty,
            "owned_tuple_named_ctor",
            &mut graph,
        );
        if let RewriteResult::Replace(ops) = &rewritten {
            assert!(
                !matches!(
                    ops.as_slice(),
                    [SpaceOperation {
                        kind: OpKind::New { owner: allocated },
                        ..
                    }] if allocated == &owner
                ),
                "an owner-qualified `Tuple<...>` leaf must not lower through \
                 the universal tuple-aggregate malloc arm",
            );
        }
    }

    /// The exact bare tuple tag is the empty tuple's `Void` representation,
    /// not a heap allocation (`rtuple.py:160-161`).
    #[test]
    fn empty_synthetic_tuple_ctor_remains_null_ref_constant() {
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config);
        let mut graph = FunctionGraph::new("empty_tuple_void");
        let result_var = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let target = CallTarget::synthetic_transparent_ctor("Tuple");
        let result_ty = ValueType::Ref(Some("Tuple".into()));
        let op = SpaceOperation {
            result: Some(result_var.clone()),
            kind: OpKind::Call {
                target: target.clone(),
                args: vec![],
                result_ty: result_ty.clone(),
            },
        };

        let RewriteResult::Replace(ops) = transformer.rewrite_op_direct_call(
            &op,
            &target,
            &[],
            &result_ty,
            "empty_tuple_void",
            &mut graph,
        ) else {
            panic!("empty Tuple aggregate must lower to its Void placeholder");
        };
        assert!(matches!(
            ops.as_slice(),
            [SpaceOperation {
                result: Some(result),
                kind: OpKind::ConstRefNull,
            }] if result == &result_var
        ));
    }

    /// PyPy parity regression guard: the qualified spellings
    /// `Result::Ok`, `Option::Some`, `std::result::Result::Err` etc.
    /// must elide identically to the bare `Ok` / `Some` / `Err`
    /// forms.  The frontend whitelist in `front::mir`
    /// already admits these multi-segment paths and records the
    /// leading segments as `owner_path`; jtransform must not reject
    /// the call based on `owner_path` being non-empty because PyPy's
    /// `rtyper`/`codewriter` collapse Ok(x) / Result::Ok(x) /
    /// std::result::Result::Ok(x) to identity at the value layer
    /// regardless of call-site spelling.
    #[test]
    fn synthetic_result_ctor_identity_accepts_qualified_spellings() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("synth_ctor_qualified");
        let arg = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let transformer = Transformer::new(&config);
        for (owner, name) in [
            (vec!["Result"], "Ok"),
            (vec!["Result"], "Err"),
            (vec!["Option"], "Some"),
            (vec!["result", "Result"], "Ok"),
            (vec!["option", "Option"], "Some"),
            (vec!["std", "result", "Result"], "Err"),
            (vec!["core", "option", "Option"], "Some"),
        ] {
            let owner_path: Vec<String> = owner.iter().map(|s| s.to_string()).collect();
            let target = CallTarget::synthetic_transparent_ctor_with_owner(owner_path, name);
            assert!(
                transformer.is_synthetic_result_option_ctor(
                    &target,
                    std::slice::from_ref(&arg),
                    &ValueType::Ref(None),
                ),
                "{owner:?}::{name} must elide identically to the bare form",
            );
        }
    }

    /// `SyntheticTransparentCtor` is also used for pyre-side unit
    /// variants like `StepResult::Continue` (Reviewer #5).  Those
    /// share the variant but must NOT elide because they are real
    /// values, not transparent wrappers.  The discriminator is the
    /// leaf name (`Ok`/`Err`/`Some` only); a non-matching name with
    /// any owner_path must still return false.
    #[test]
    fn synthetic_result_ctor_identity_rejects_unit_variant_with_owner() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("synth_ctor_unit_variant");
        let arg = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let transformer = Transformer::new(&config);
        let target = CallTarget::synthetic_transparent_ctor_with_owner(
            vec!["StepResult".to_string()],
            "Continue",
        );
        assert!(!transformer.is_synthetic_result_option_ctor(
            &target,
            &[arg],
            &ValueType::Ref(None),
        ));
    }

    /// Reject names not in the narrow allow-list and reject name-only
    /// matching for qualified paths. The frontend maps approved qualified
    /// constructors to `SyntheticTransparentCtor` with the final segment.
    #[test]
    fn synthetic_result_ctor_identity_rejects_other_names() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("synth_ctor_other_names");
        let arg = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let transformer = Transformer::new(&config);
        assert!(!transformer.is_synthetic_result_option_ctor(
            &CallTarget::function_path(["Result", "Ok"]),
            std::slice::from_ref(&arg),
            &ValueType::Ref(None),
        ));
        assert!(!transformer.is_synthetic_result_option_ctor(
            &CallTarget::synthetic_transparent_ctor("Foo"),
            std::slice::from_ref(&arg),
            &ValueType::Ref(None),
        ));
        assert!(transformer.is_synthetic_result_option_ctor(
            &CallTarget::synthetic_transparent_ctor("Err"),
            std::slice::from_ref(&arg),
            &ValueType::Ref(None),
        ));
        assert!(transformer.is_synthetic_result_option_ctor(
            &CallTarget::synthetic_transparent_ctor("Some"),
            &[arg],
            &ValueType::Ref(None),
        ));
    }

    #[test]
    fn map_user_oopspec_dict_lookup() {
        use majit_ir::descr::OopSpecIndex;
        assert_eq!(
            super::map_user_oopspec_to_index("ordereddict.lookup(d, key, hash, flag)"),
            OopSpecIndex::DictLookup
        );
        assert_eq!(
            super::map_user_oopspec_to_index("ordereddict.lookup"),
            OopSpecIndex::DictLookup
        );
        assert_eq!(
            super::map_user_oopspec_to_index("dict.lookup"),
            OopSpecIndex::DictLookup
        );
        assert_eq!(
            super::map_user_oopspec_to_index("dict.setitem"),
            OopSpecIndex::None
        );
    }

    fn stroruni_copy_contents_args(
        ptr_lltype: crate::translator::rtyper::lltypesystem::lltype::LowLevelType,
    ) -> Vec<crate::flowspace::model::Variable> {
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::rtyper::variable_with_lltype;
        vec![
            variable_with_lltype("src", ptr_lltype.clone()),
            variable_with_lltype("dst", ptr_lltype),
            variable_with_lltype("srcstart", LowLevelType::Signed),
            variable_with_lltype("dststart", LowLevelType::Signed),
            variable_with_lltype("length", LowLevelType::Signed),
        ]
    }

    /// The `copy_contents` arm returns before touching the residual-call
    /// machinery, so its tests only need placeholder `graph` / `target` /
    /// `result_ty` values to satisfy the shared `_handle_stroruni_call`
    /// signature.
    fn stroruni_call_dummy_extras() -> (FunctionGraph, CallTarget, ValueType) {
        (
            FunctionGraph::new("stroruni"),
            CallTarget::function_path(["copy_contents"]),
            ValueType::Ref(None),
        )
    }

    /// jtransform.py:2079-2086 — `stroruni.copy_contents` with a `Ptr(STR)`
    /// first argument lowers to a single `copystrcontent` blackhole op
    /// carrying the five call args and a void result.
    #[test]
    fn stroruni_copy_contents_str_lowers_to_copystrcontent() {
        use crate::translator::rtyper::lltypesystem::rstr::STRPTR;
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config);
        let args = stroruni_copy_contents_args(STRPTR.clone());
        let op = SpaceOperation {
            result: None,
            kind: OpKind::Live,
        };
        let (mut graph, target, result_ty) = stroruni_call_dummy_extras();
        let result = transformer
            ._handle_stroruni_call(
                "stroruni.copy_contents",
                &op,
                &target,
                &args,
                &result_ty,
                "stroruni",
                &mut graph,
            )
            .expect("copy_contents must be handled");
        let RewriteResult::Replace(ops) = result else {
            panic!("expected Replace");
        };
        assert_eq!(ops.len(), 1);
        let OpKind::LoweredBlackholeOp {
            opname,
            args: bh_args,
        } = &ops[0].kind
        else {
            panic!("expected LoweredBlackholeOp, got {:?}", ops[0].kind);
        };
        assert_eq!(opname, "copystrcontent");
        assert_eq!(bh_args.len(), 5);
        assert!(ops[0].result.is_none());
    }

    /// jtransform.py:2082-2083 — the `Ptr(UNICODE)` first argument selects
    /// the `copyunicodecontent` spelling instead.
    #[test]
    fn stroruni_copy_contents_unicode_lowers_to_copyunicodecontent() {
        use crate::translator::rtyper::lltypesystem::rstr::UNICODEPTR;
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config);
        let args = stroruni_copy_contents_args(UNICODEPTR.clone());
        let op = SpaceOperation {
            result: None,
            kind: OpKind::Live,
        };
        let (mut graph, target, result_ty) = stroruni_call_dummy_extras();
        let result = transformer
            ._handle_stroruni_call(
                "stroruni.copy_contents",
                &op,
                &target,
                &args,
                &result_ty,
                "stroruni",
                &mut graph,
            )
            .expect("copy_contents must be handled");
        let RewriteResult::Replace(ops) = result else {
            panic!("expected Replace");
        };
        let OpKind::LoweredBlackholeOp { opname, .. } = &ops[0].kind else {
            panic!("expected LoweredBlackholeOp");
        };
        assert_eq!(opname, "copyunicodecontent");
    }

    /// jtransform.py:2074 — a `Ptr(BYTEARRAY)` first argument is unsupported
    /// (`raise NotSupported`); pyre returns `None` so the call falls through
    /// to the residual-call path instead of misclassifying it as `STR`.
    #[test]
    fn stroruni_copy_contents_bytearray_falls_through() {
        use crate::translator::rtyper::lltypesystem::rbytearray::BYTEARRAYPTR;
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config);
        let args = stroruni_copy_contents_args(BYTEARRAYPTR.clone());
        let op = SpaceOperation {
            result: None,
            kind: OpKind::Live,
        };
        let (mut graph, target, result_ty) = stroruni_call_dummy_extras();
        assert!(
            transformer
                ._handle_stroruni_call(
                    "stroruni.copy_contents",
                    &op,
                    &target,
                    &args,
                    &result_ty,
                    "stroruni",
                    &mut graph,
                )
                .is_none()
        );
    }

    /// jtransform.py:2059-2072, 2124-2128 — `stroruni.concat` on a `Ptr(STR)`
    /// operand lowers to a residual call carrying the `StrConcat` oopspecindex
    /// (`OS_STR_CONCAT`).
    #[test]
    fn stroruni_concat_lowers_to_str_concat_residual_call() {
        use crate::call::CallControl;
        use crate::translator::rtyper::lltypesystem::rstr::STRPTR;

        let mut cc = CallControl::new();
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);

        let mut graph = FunctionGraph::new("concat");
        let args = stroruni_copy_contents_args(STRPTR.clone());
        let result_ty = ValueType::Ref(None);
        let target = CallTarget::function_path(["ll_strconcat"]);
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "result".into(),
                    ty: result_ty.clone(),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let op = SpaceOperation {
            result: Some(result_var),
            kind: OpKind::Call {
                target: target.clone(),
                args: args.clone(),
                result_ty: result_ty.clone(),
            },
        };
        let rewritten = transformer
            ._handle_stroruni_call(
                "stroruni.concat",
                &op,
                &target,
                &args,
                &result_ty,
                "concat",
                &mut graph,
            )
            .expect("stroruni.concat must be handled");
        let RewriteResult::Replace(ops) = rewritten else {
            panic!("expected Replace");
        };
        assert!(
            ops.iter()
                .any(|o| matches!(o.kind, OpKind::CallResidual { .. })),
            "expected a CallResidual, got {ops:?}"
        );
        assert!(
            transformer
                .notes
                .iter()
                .any(|n| n.detail.contains("StrConcat")),
            "expected an oopspec StrConcat note, got {:?}",
            transformer.notes
        );
    }

    /// jtransform.py:2087-2128 — `stroruni.equal` emits the main OS_STR_EQUAL
    /// residual call (via the common tail) even though the seven
    /// OS_STREQ_*/OS_UNIEQ_* extra-helper side-registrations remain blocked on
    /// the unported str.eq_* host bodies.
    #[test]
    fn stroruni_equal_lowers_to_str_equal_residual_call() {
        use crate::call::CallControl;
        use crate::translator::rtyper::lltypesystem::rstr::STRPTR;

        let mut cc = CallControl::new();
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);

        let mut graph = FunctionGraph::new("equal");
        let args = stroruni_copy_contents_args(STRPTR.clone());
        let result_ty = ValueType::Int;
        let target = CallTarget::function_path(["ll_streq"]);
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "result".into(),
                    ty: result_ty.clone(),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let op = SpaceOperation {
            result: Some(result_var),
            kind: OpKind::Call {
                target: target.clone(),
                args: args.clone(),
                result_ty: result_ty.clone(),
            },
        };
        let rewritten = transformer
            ._handle_stroruni_call(
                "stroruni.equal",
                &op,
                &target,
                &args,
                &result_ty,
                "equal",
                &mut graph,
            )
            .expect("stroruni.equal must emit its OS_STR_EQUAL residual call");
        let RewriteResult::Replace(ops) = rewritten else {
            panic!("expected Replace");
        };
        assert!(
            ops.iter()
                .any(|o| matches!(o.kind, OpKind::CallResidual { .. })),
            "expected a CallResidual, got {ops:?}"
        );
        assert!(
            transformer
                .notes
                .iter()
                .any(|n| n.detail.contains("StrEqual")),
            "expected an oopspec StrEqual note, got {:?}",
            transformer.notes
        );
    }

    /// jtransform.py:2193-2195 — `rgc.ll_shrink_array` lowers to a residual
    /// call carrying the `ShrinkArray` oopspecindex and `CanRaise` effect,
    /// so `handle_residual_call` appends a trailing `-live-`.
    #[test]
    fn rgc_ll_shrink_array_lowers_to_shrink_array_residual_call() {
        use crate::call::CallControl;
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::lltypesystem::rstr::STRPTR;
        use crate::translator::rtyper::rtyper::variable_with_lltype;

        let mut cc = CallControl::new();
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);

        let mut graph = FunctionGraph::new("shrink");
        let p = variable_with_lltype("p", STRPTR.clone());
        let smaller = variable_with_lltype("smallerlength", LowLevelType::Signed);
        let args = vec![p, smaller];
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "result".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let result_ty = ValueType::Ref(None);
        let target = CallTarget::function_path(["ll_shrink_array"]);
        let op = SpaceOperation {
            result: Some(result_var),
            kind: OpKind::Call {
                target: target.clone(),
                args: args.clone(),
                result_ty: result_ty.clone(),
            },
        };
        let rewritten = transformer
            ._handle_rgc_call(
                "rgc.ll_shrink_array",
                &op,
                &target,
                &args,
                &result_ty,
                "shrink",
                &mut graph,
            )
            .expect("ll_shrink_array must be handled");
        let RewriteResult::Replace(ops) = rewritten else {
            panic!("expected Replace");
        };
        assert!(
            ops.iter()
                .any(|o| matches!(o.kind, OpKind::CallResidual { .. })),
            "expected a CallResidual, got {ops:?}"
        );
        // CanRaise → trailing -live-.
        assert!(
            matches!(ops.last().map(|o| &o.kind), Some(OpKind::Live)),
            "CanRaise oopspec call must append a trailing -live-"
        );
    }

    /// Other `rgc.*` spellings are not ported and fall through to the
    /// residual-call path.
    #[test]
    fn rgc_unported_spellings_fall_through() {
        use crate::call::CallControl;
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        use crate::translator::rtyper::rtyper::variable_with_lltype;

        let mut cc = CallControl::new();
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
        let mut graph = FunctionGraph::new("rgc_other");
        let args = vec![variable_with_lltype("x", LowLevelType::Signed)];
        let op = SpaceOperation {
            result: None,
            kind: OpKind::Live,
        };
        assert!(
            transformer
                ._handle_rgc_call(
                    "rgc.something_else",
                    &op,
                    &CallTarget::function_path(["something_else"]),
                    &args,
                    &ValueType::Void,
                    "rgc_other",
                    &mut graph,
                )
                .is_none()
        );
    }

    //
    // RPython upstream: `jtransform.py:538-553 handle_regular_indirect_
    // call` emits `[-live-, int_guard_value, residual_call +
    // IndirectCallTargets, -live-]`; `assembler.py:208-209` collects
    // the candidate jitcodes into `indirectcalltargets`.  These two
    // tests anchor the RPython-orthodox post-jtransform shape so any
    // future change to `lower_indirect_calls` / `Transformer::transform`
    // surfaces here rather than reaching the assembler with a wrong
    // op sequence.

    /// Build an impl-method graph with a single `Input { ty: Ref }`
    /// representing the receiver `self` — mirrors the one-arg signature
    /// `fn run(&self)` that `FunctionReprBase.call` feeds into
    /// `indirect_call(funcptr, self, c_graphs)` (`rpbc.py:207-217`).
    fn build_handler_run_impl_graph(name: &str) -> FunctionGraph {
        let mut graph = FunctionGraph::new(name);
        graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "self".to_string(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, None);
        graph
    }

    /// Build a graph that calls `receiver.run()` on a `dyn Handler`
    /// receiver with two registered impls, run the rtyper-equivalent
    /// `lower_indirect_calls` pass and then `Transformer::transform`,
    /// and assert the post-jtransform sequence is exactly
    /// `[VtableMethodPtr, Live, GuardValue{kind='i'},
    ///   CallResidual{funcptr=Value(_), indirect_targets=Some}, Live]`.
    /// RPython `jtransform.py:410-412 + 538-553` orthodox port parity.
    #[test]
    fn lower_indirect_call_op_emit_order() {
        use crate::call::CallControl;
        use crate::translator::rtyper::legacy_annotator::annotate;
        use crate::translator::rtyper::legacy_resolve::resolve_types;
        use crate::translator::rtyper::rpbc::lower_indirect_calls;

        let mut cc = CallControl::new();
        cc.register_trait_method(
            "run",
            Some("Handler"),
            "A",
            build_handler_run_impl_graph("A::run"),
        );
        cc.register_trait_method(
            "run",
            Some("Handler"),
            "B",
            build_handler_run_impl_graph("B::run"),
        );
        cc.find_all_graphs_for_tests();

        let mut graph = FunctionGraph::new("outer");
        let receiver_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "handler".to_string(),
                    ty: ValueType::Unknown,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::indirect("Handler", "run"),
                args: vec![receiver_var],
                result_ty: ValueType::Void,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        annotate(&graph);
        resolve_types(&graph);
        lower_indirect_calls(&mut graph, &cc);

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
        let result = transformer.transform(&graph);

        let ops = &result.graph.block(graph.startblock).operations;
        let post_input = &ops[1..];
        assert!(
            matches!(post_input[0].kind, OpKind::VtableMethodPtr { .. }),
            "expected VtableMethodPtr first, got {:?}",
            post_input[0].kind
        );
        assert!(
            matches!(post_input[1].kind, OpKind::Live),
            "expected Live second, got {:?}",
            post_input[1].kind
        );
        match &post_input[2].kind {
            OpKind::GuardValue { kind_char: 'i', .. } => {}
            other => panic!("expected GuardValue kind='i', got {other:?}"),
        }
        match &post_input[3].kind {
            OpKind::CallResidual {
                funcptr: CallFuncPtr::Value(_),
                indirect_targets: Some(t),
                ..
            } => assert_eq!(t.lst.len(), 2, "both impls should be candidates"),
            other => panic!(
                "expected CallResidual with runtime funcptr + indirect_targets, got {other:?}"
            ),
        }
        // jtransform.py:547 handle_residual_call(..., may_call_jitcodes=True)
        // forces a trailing `-live-`.
        assert!(
            matches!(post_input[4].kind, OpKind::Live),
            "expected trailing Live, got {:?}",
            post_input[4].kind
        );
    }

    /// End-to-end smoke: after the rtyper-equivalent `lower_indirect_calls`
    /// pass + `Transformer::transform`, the `CallResidual.indirect_targets`
    /// payload carries exactly one `JitCodeHandle` per candidate impl
    /// (each shell allocated by `CallControl::get_jitcode`). The assembler
    /// later merges these handles into `Assembler.indirectcalltargets`
    /// (RPython `assembler.py:208-209`).
    #[test]
    fn indirectcalltargets_reach_call_residual_payload() {
        use crate::call::CallControl;
        use crate::translator::rtyper::legacy_annotator::annotate;
        use crate::translator::rtyper::legacy_resolve::resolve_types;
        use crate::translator::rtyper::rpbc::lower_indirect_calls;

        let mut cc = CallControl::new();
        cc.register_trait_method(
            "run",
            Some("Handler"),
            "A",
            build_handler_run_impl_graph("A::run"),
        );
        cc.register_trait_method(
            "run",
            Some("Handler"),
            "B",
            build_handler_run_impl_graph("B::run"),
        );
        cc.find_all_graphs_for_tests();

        let mut graph = FunctionGraph::new("outer");
        let receiver_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "handler".to_string(),
                    ty: ValueType::Unknown,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::indirect("Handler", "run"),
                args: vec![receiver_var],
                result_ty: ValueType::Void,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        annotate(&graph);
        resolve_types(&graph);
        lower_indirect_calls(&mut graph, &cc);

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

        assert_eq!(residual.lst.len(), 2);
        // `IndirectCallTargets.lst` carries shell handles returned by
        // `CallControl::get_jitcode()`. RPython appends/indices jitcodes
        // only after `transform_graph_to_jitcode(...)` completes, so the
        // candidate handles must still be unindexed at this stage.
        assert!(
            residual.lst.iter().all(|h| h.try_index().is_none()),
            "indirect target shells should not have final all_jitcodes indices yet"
        );
    }

    // ── Kind matrix: `indirect_regular_call_{r,ir,irf}_{i,r,f,v}` ────
    //
    // RPython upstream: `test_jtransform.py:340-367` parameterization +
    // `test_jtransform.py:447-484 indirect_regular_call_test`.  Each
    // test builds a `receiver.m(extras...)` site where `extras` covers
    // the arg kind signature (`r`, `ir`, `irf`) and `result_ty` covers
    // the result kind (`i`, `r`, `f`, `v`).  The emitted `CallResidual`
    // must split args into `(args_i, args_r, args_f)` by kind — with
    // the receiver always landing in `args_r` — and carry `result_kind`
    // matching the call's `result_ty`.  Receiver-in-args mirrors RPython
    // `rpbc.py:1195-1208 MethodsPBCRepr.redispatch_call`.

    /// Runs the full rtyper-equivalent + jtransform pipeline for a
    /// `receiver.m(extras...)` dyn-Trait call and asserts the emitted
    /// op sequence is exactly
    /// `[VtableMethodPtr, Live, GuardValue{'i'}, CallResidual, Live]`,
    /// with the CallResidual's arg-kind distribution matching `expect`
    /// and `result_kind` matching `expect_res_kind`.  Two impls are
    /// registered so `CallKind::Regular` is selected.
    #[cfg(test)]
    fn check_indirect_regular_call_kind(
        extras: &[ValueType],
        result_ty: ValueType,
        expect: (usize, usize, usize),
        expect_res_kind: char,
    ) {
        use crate::call::CallControl;
        use crate::translator::rtyper::legacy_annotator::annotate;
        use crate::translator::rtyper::legacy_resolve::resolve_types;
        use crate::translator::rtyper::rpbc::lower_indirect_calls;

        let build_impl = |name: &str| {
            let mut g = FunctionGraph::new(name);
            g.push_op_var(
                g.startblock,
                OpKind::Input {
                    name: "self".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
            for (i, ty) in extras.iter().enumerate() {
                g.push_op_var(
                    g.startblock,
                    OpKind::Input {
                        name: format!("a{i}"),
                        ty: ty.clone(),
                        class_root: None,
                    },
                    true,
                )
                .unwrap();
            }
            g.set_return(g.startblock, None);
            g
        };

        // The indirect branch of `getcalldescr(op)` validates that the
        // caller's `result_type` matches the witness impl's declared
        // return type (`call.rs` indirect-arm signature check).  Stamp the
        // matching return type onto the impl graphs so the non-void
        // kind-matrix cases resolve without panicking.
        let return_str = match result_ty {
            ValueType::Int | ValueType::State => "i64",
            ValueType::Unsigned => "u64",
            ValueType::Ref(_) | ValueType::Str | ValueType::Unknown => "String",
            ValueType::Float => "f64",
            ValueType::Void => "",
            ValueType::Bool => "bool",
            ValueType::Int128 => "i128",
            ValueType::UInt128 => "u128",
        };
        let build_witness = |name: &str| {
            let g = build_impl(name);
            if return_str.is_empty() {
                g
            } else {
                g.with_return_type(return_str)
            }
        };
        let mut cc = CallControl::new();
        cc.register_trait_method("m", Some("T"), "A", build_witness("A::m"));
        cc.register_trait_method("m", Some("T"), "B", build_witness("B::m"));
        cc.find_all_graphs_for_tests();

        let mut graph = FunctionGraph::new("outer");
        let receiver_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "self".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let mut args_vars: Vec<crate::flowspace::model::Variable> = vec![receiver_var];
        for (i, ty) in extras.iter().enumerate() {
            args_vars.push(
                graph
                    .push_op_var(
                        graph.startblock,
                        OpKind::Input {
                            name: format!("a{i}"),
                            ty: ty.clone(),
                            class_root: None,
                        },
                        true,
                    )
                    .unwrap(),
            );
        }
        let has_result = !matches!(result_ty, ValueType::Void);
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::indirect("T", "m"),
                args: args_vars,
                result_ty,
            },
            has_result,
        );
        graph.set_return(graph.startblock, None);

        annotate(&graph);
        resolve_types(&graph);
        lower_indirect_calls(&mut graph, &cc);

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
        let result = transformer.transform(&graph);

        let ops = &result.graph.block(graph.startblock).operations;
        let post_input: Vec<_> = ops
            .iter()
            .filter(|op| !matches!(&op.kind, OpKind::Input { .. }))
            .collect();

        assert!(
            matches!(post_input[0].kind, OpKind::VtableMethodPtr { .. }),
            "expected VtableMethodPtr first, got {:?}",
            post_input[0].kind,
        );
        assert!(
            matches!(post_input[1].kind, OpKind::Live),
            "expected Live second, got {:?}",
            post_input[1].kind,
        );
        assert!(
            matches!(
                post_input[2].kind,
                OpKind::GuardValue { kind_char: 'i', .. }
            ),
            "expected GuardValue kind_char='i', got {:?}",
            post_input[2].kind,
        );
        match &post_input[3].kind {
            OpKind::CallResidual {
                funcptr: CallFuncPtr::Value(_),
                args_i,
                args_r,
                args_f,
                result_kind,
                indirect_targets: Some(t),
                ..
            } => {
                assert_eq!(args_i.len(), expect.0, "args_i count");
                assert_eq!(args_r.len(), expect.1, "args_r count (receiver counted)");
                assert_eq!(args_f.len(), expect.2, "args_f count");
                assert_eq!(*result_kind, expect_res_kind, "result_kind");
                assert_eq!(t.lst.len(), 2, "both impls in family");
            }
            other => {
                panic!("expected CallResidual with Value funcptr + indirect_targets, got {other:?}")
            }
        }
        assert!(
            matches!(post_input[4].kind, OpKind::Live),
            "expected trailing Live (may_call_jitcodes=true), got {:?}",
            post_input[4].kind,
        );
    }

    #[test]
    fn indirect_regular_call_r_i() {
        check_indirect_regular_call_kind(&[], ValueType::Int, (0, 1, 0), 'i');
    }
    #[test]
    fn indirect_regular_call_r_r() {
        check_indirect_regular_call_kind(&[], ValueType::Ref(None), (0, 1, 0), 'r');
    }
    #[test]
    fn indirect_regular_call_r_f() {
        check_indirect_regular_call_kind(&[], ValueType::Float, (0, 1, 0), 'f');
    }
    #[test]
    fn indirect_regular_call_r_v() {
        check_indirect_regular_call_kind(&[], ValueType::Void, (0, 1, 0), 'v');
    }

    #[test]
    fn indirect_regular_call_ir_i() {
        check_indirect_regular_call_kind(&[ValueType::Int], ValueType::Int, (1, 1, 0), 'i');
    }
    #[test]
    fn indirect_regular_call_ir_r() {
        check_indirect_regular_call_kind(&[ValueType::Int], ValueType::Ref(None), (1, 1, 0), 'r');
    }
    #[test]
    fn indirect_regular_call_ir_f() {
        check_indirect_regular_call_kind(&[ValueType::Int], ValueType::Float, (1, 1, 0), 'f');
    }
    #[test]
    fn indirect_regular_call_ir_v() {
        check_indirect_regular_call_kind(&[ValueType::Int], ValueType::Void, (1, 1, 0), 'v');
    }

    #[test]
    fn indirect_regular_call_irf_i() {
        check_indirect_regular_call_kind(
            &[ValueType::Int, ValueType::Float],
            ValueType::Int,
            (1, 1, 1),
            'i',
        );
    }
    #[test]
    fn indirect_regular_call_irf_r() {
        check_indirect_regular_call_kind(
            &[ValueType::Int, ValueType::Float],
            ValueType::Ref(None),
            (1, 1, 1),
            'r',
        );
    }
    #[test]
    fn indirect_regular_call_irf_f() {
        check_indirect_regular_call_kind(
            &[ValueType::Int, ValueType::Float],
            ValueType::Float,
            (1, 1, 1),
            'f',
        );
    }
    #[test]
    fn indirect_regular_call_irf_v() {
        check_indirect_regular_call_kind(
            &[ValueType::Int, ValueType::Float],
            ValueType::Void,
            (1, 1, 1),
            'v',
        );
    }

    // ── Kind matrix: `indirect_residual_call_{r,ir,irf}_{i,r,f,v}` ───
    //
    // RPython upstream: `test_jtransform.py:420-445 indirect_residual_call_test`.
    // `handle_residual_indirect_call` is an alias for the regular
    // residual-call path (`jtransform.py:536`): no `int_guard_value` guard
    // and no `IndirectCallTargets` sidecar; only `[residual_call_*, -live-]`.
    // On the pyre side we additionally retain the `VtableMethodPtr` op
    // emitted by `lower_indirect_calls` — the rtyper-equivalent layer runs
    // unconditionally and provides the runtime funcptr `Variable` consumed
    // by the `CallResidual.funcptr` `CallFuncPtr::Value` operand.

    /// Same skeleton as `check_indirect_regular_call_kind` but drops
    /// `find_all_graphs_for_tests`, so `candidate_graphs` stays empty
    /// and `guess_call_kind(op)` falls through to `CallKind::Residual`
    /// via the `graphs_from(op)` call.py:137-139 fall-through.
    /// Emit is `[VtableMethodPtr, CallResidual{indirect_targets: None},
    /// Live?]` — the trailing `-live-` appears only when the descriptor's
    /// `can_raise` is true, which is the default for non-elidable
    /// family calls.
    #[cfg(test)]
    fn check_indirect_residual_call_kind(
        extras: &[ValueType],
        result_ty: ValueType,
        expect: (usize, usize, usize),
        expect_res_kind: char,
    ) {
        use crate::call::CallControl;
        use crate::translator::rtyper::legacy_annotator::annotate;
        use crate::translator::rtyper::legacy_resolve::resolve_types;
        use crate::translator::rtyper::rpbc::lower_indirect_calls;

        let build_impl = |name: &str| {
            let mut g = FunctionGraph::new(name);
            g.push_op_var(
                g.startblock,
                OpKind::Input {
                    name: "self".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
            for (i, ty) in extras.iter().enumerate() {
                g.push_op_var(
                    g.startblock,
                    OpKind::Input {
                        name: format!("a{i}"),
                        ty: ty.clone(),
                        class_root: None,
                    },
                    true,
                )
                .unwrap();
            }
            g.set_return(g.startblock, None);
            g
        };

        let return_str = match result_ty {
            ValueType::Int | ValueType::State => "i64",
            ValueType::Unsigned => "u64",
            ValueType::Ref(_) | ValueType::Str | ValueType::Unknown => "String",
            ValueType::Float => "f64",
            ValueType::Void => "",
            ValueType::Bool => "bool",
            ValueType::Int128 => "i128",
            ValueType::UInt128 => "u128",
        };
        let build_witness = |name: &str| {
            let g = build_impl(name);
            if return_str.is_empty() {
                g
            } else {
                g.with_return_type(return_str)
            }
        };
        let mut cc = CallControl::new();
        cc.register_trait_method("m", Some("T"), "A", build_witness("A::m"));
        cc.register_trait_method("m", Some("T"), "B", build_witness("B::m"));
        // NOTE: intentionally *no* `find_all_graphs_for_tests` — that keeps
        // `candidate_graphs` empty so `guess_call_kind(op)` classifies
        // this call as `CallKind::Residual` via the call.py:137-139
        // `graphs_from(op) is None` fall-through.

        let mut graph = FunctionGraph::new("outer");
        let receiver_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "self".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let mut args_vars: Vec<crate::flowspace::model::Variable> = vec![receiver_var];
        for (i, ty) in extras.iter().enumerate() {
            args_vars.push(
                graph
                    .push_op_var(
                        graph.startblock,
                        OpKind::Input {
                            name: format!("a{i}"),
                            ty: ty.clone(),
                            class_root: None,
                        },
                        true,
                    )
                    .unwrap(),
            );
        }
        let has_result = !matches!(result_ty, ValueType::Void);
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::indirect("T", "m"),
                args: args_vars,
                result_ty,
            },
            has_result,
        );
        graph.set_return(graph.startblock, None);

        annotate(&graph);
        resolve_types(&graph);
        lower_indirect_calls(&mut graph, &cc);

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
        let result = transformer.transform(&graph);

        let ops = &result.graph.block(graph.startblock).operations;
        let post_input: Vec<_> = ops
            .iter()
            .filter(|op| !matches!(&op.kind, OpKind::Input { .. }))
            .collect();

        assert!(
            matches!(post_input[0].kind, OpKind::VtableMethodPtr { .. }),
            "expected VtableMethodPtr first, got {:?}",
            post_input[0].kind,
        );
        // Residual path must NOT emit the [Live, GuardValue] prefix —
        // upstream `jtransform.py:536` aliases
        // `handle_residual_indirect_call = handle_residual_call`.
        assert!(
            !post_input
                .iter()
                .any(|op| matches!(op.kind, OpKind::GuardValue { .. })),
            "residual path must not emit int_guard_value, got ops: {:?}",
            post_input.iter().map(|op| &op.kind).collect::<Vec<_>>(),
        );
        match &post_input[1].kind {
            OpKind::CallResidual {
                funcptr: CallFuncPtr::Value(_),
                args_i,
                args_r,
                args_f,
                result_kind,
                indirect_targets: None,
                ..
            } => {
                assert_eq!(args_i.len(), expect.0, "args_i count");
                assert_eq!(args_r.len(), expect.1, "args_r count (receiver counted)");
                assert_eq!(args_f.len(), expect.2, "args_f count");
                assert_eq!(*result_kind, expect_res_kind, "result_kind");
            }
            other => panic!(
                "expected CallResidual with Value funcptr + indirect_targets=None, got {other:?}"
            ),
        }
    }

    #[test]
    fn indirect_residual_call_r_i() {
        check_indirect_residual_call_kind(&[], ValueType::Int, (0, 1, 0), 'i');
    }
    #[test]
    fn indirect_residual_call_r_r() {
        check_indirect_residual_call_kind(&[], ValueType::Ref(None), (0, 1, 0), 'r');
    }
    #[test]
    fn indirect_residual_call_r_f() {
        check_indirect_residual_call_kind(&[], ValueType::Float, (0, 1, 0), 'f');
    }
    #[test]
    fn indirect_residual_call_r_v() {
        check_indirect_residual_call_kind(&[], ValueType::Void, (0, 1, 0), 'v');
    }

    #[test]
    fn indirect_residual_call_ir_i() {
        check_indirect_residual_call_kind(&[ValueType::Int], ValueType::Int, (1, 1, 0), 'i');
    }
    #[test]
    fn indirect_residual_call_ir_r() {
        check_indirect_residual_call_kind(&[ValueType::Int], ValueType::Ref(None), (1, 1, 0), 'r');
    }
    #[test]
    fn indirect_residual_call_ir_f() {
        check_indirect_residual_call_kind(&[ValueType::Int], ValueType::Float, (1, 1, 0), 'f');
    }
    #[test]
    fn indirect_residual_call_ir_v() {
        check_indirect_residual_call_kind(&[ValueType::Int], ValueType::Void, (1, 1, 0), 'v');
    }

    #[test]
    fn indirect_residual_call_irf_i() {
        check_indirect_residual_call_kind(
            &[ValueType::Int, ValueType::Float],
            ValueType::Int,
            (1, 1, 1),
            'i',
        );
    }
    #[test]
    fn indirect_residual_call_irf_r() {
        check_indirect_residual_call_kind(
            &[ValueType::Int, ValueType::Float],
            ValueType::Ref(None),
            (1, 1, 1),
            'r',
        );
    }
    #[test]
    fn indirect_residual_call_irf_f() {
        check_indirect_residual_call_kind(
            &[ValueType::Int, ValueType::Float],
            ValueType::Float,
            (1, 1, 1),
            'f',
        );
    }
    #[test]
    fn indirect_residual_call_irf_v() {
        check_indirect_residual_call_kind(
            &[ValueType::Int, ValueType::Float],
            ValueType::Void,
            (1, 1, 1),
            'v',
        );
    }

    /// `rpython/jit/codewriter/jtransform.py:599-606` — when a
    /// `hint(arg, promote=True, promote_string=True)` carries both
    /// flags, the rewrite drops one based on whether the arg's
    /// `concretetype` is `Ptr(STR)`.  Pyre's `value_kind` is too
    /// coarse to make that distinction (every pointer maps to `'r'`),
    /// so `PromoteOrString` defaults to the plain `<kind>_guard_value`
    /// path — `ref_guard_value` is safe for every Ref, including
    /// non-string pointers.  Users who need the value-equality
    /// `str_guard_value` shape invoke `hint_promote_string(x)`
    /// explicitly.
    #[test]
    fn transform_graph_promote_or_string_picks_ref_guard_value_for_ref_arg() {
        let mut graph = FunctionGraph::new("demo");
        let v_var = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, v_var.clone());
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["hint_promote_or_string"]),
                args: vec![v_var],
                result_ty: ValueType::Ref(None),
            },
            false,
        );
        graph.set_return(graph.startblock, None);

        let result = transform_graph(&graph, &GraphTransformConfig::default());
        let ops = &result.graph.block(graph.startblock).operations;
        assert_eq!(ops.len(), 2);
        assert!(matches!(ops[0].kind, OpKind::Live));
        match &ops[1].kind {
            OpKind::GuardValue { kind_char, .. } => {
                assert_eq!(
                    *kind_char, 'r',
                    "Ref arg must default to ref_guard_value (no Ptr(STR) info)"
                );
            }
            other => panic!("expected GuardValue, got {other:?}"),
        }
    }

    #[test]
    fn transform_graph_promote_or_string_picks_int_guard_value_for_int_arg() {
        let mut graph = FunctionGraph::new("demo");
        let v_var = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, v_var.clone());
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["hint_promote_or_string"]),
                args: vec![v_var.clone()],
                result_ty: ValueType::Int,
            },
            false,
        );
        graph.set_return(graph.startblock, None);

        // value_kind defaults to 'r' without an entry in type_state;
        // seed `v` as `Signed` so the dual-hint arm sees an Int arg
        // and routes through the plain `<kind>_guard_value` path
        // instead of the str_guard_value helper chain.
        FunctionGraph::set_concretetype_of_inline(&v_var, ConcreteType::Signed);
        let config = GraphTransformConfig::default();
        let result = Transformer::new(&config).transform(&graph);
        let ops = &result.graph.block(graph.startblock).operations;
        assert!(matches!(ops[0].kind, OpKind::Live));
        match &ops[1].kind {
            OpKind::GuardValue { kind_char, .. } => {
                assert_eq!(
                    *kind_char, 'i',
                    "Int arg must route through the int_guard_value path"
                );
            }
            other => panic!("expected GuardValue, got {other:?}"),
        }
    }

    /// `promote_string` lowers a `W_UnicodeObject` ref through the shared
    /// `<kind>_guard_value` family: pyre has no `Ptr(rstr.STR)` to
    /// value-compare, so the ref operand collapses to `r_guard_value`
    /// (`rpython/jit/codewriter/jtransform.py:615-631`).
    #[test]
    fn transform_graph_promote_string_emits_ref_guard_value() {
        let mut graph = FunctionGraph::new("demo");
        let v_var = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, v_var.clone());
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["hint_promote_string"]),
                args: vec![v_var],
                result_ty: ValueType::Ref(None),
            },
            false,
        );
        graph.set_return(graph.startblock, None);

        let result = transform_graph(&graph, &GraphTransformConfig::default());
        let ops = &result.graph.block(graph.startblock).operations;
        assert_eq!(ops.len(), 2);
        assert!(matches!(ops[0].kind, OpKind::Live));
        match &ops[1].kind {
            OpKind::GuardValue { kind_char, .. } => {
                assert_eq!(
                    *kind_char, 'r',
                    "string promote on a ref must emit r_guard_value"
                );
            }
            other => panic!("expected GuardValue, got {other:?}"),
        }
    }

    /// `promote_unicode` shares the `PromoteString` lowering — the
    /// `rstr.UNICODE` value-equality guard has no `W_UnicodeObject`
    /// counterpart, so the ref operand emits `r_guard_value`
    /// (`rpython/jit/codewriter/jtransform.py:632-648`).
    #[test]
    fn transform_graph_promote_unicode_emits_ref_guard_value() {
        let mut graph = FunctionGraph::new("demo");
        let v_var = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, v_var.clone());
        graph.push_op_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["hint_promote_unicode"]),
                args: vec![v_var],
                result_ty: ValueType::Ref(None),
            },
            false,
        );
        graph.set_return(graph.startblock, None);

        let result = transform_graph(&graph, &GraphTransformConfig::default());
        let ops = &result.graph.block(graph.startblock).operations;
        assert_eq!(ops.len(), 2);
        assert!(matches!(ops[0].kind, OpKind::Live));
        match &ops[1].kind {
            OpKind::GuardValue { kind_char, .. } => {
                assert_eq!(
                    *kind_char, 'r',
                    "unicode promote on a ref must emit r_guard_value"
                );
            }
            other => panic!("expected GuardValue, got {other:?}"),
        }
    }

    /// `list.int_len(l)` lowers to a single `getfield_gc_i(l,
    /// int_items.len)` (`do_resizable_list_len`).
    #[test]
    fn handle_list_call_int_len_lowers_to_getfield_len() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("list_int_len");
        let l = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let result = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let op = SpaceOperation {
            result: Some(result.clone()),
            kind: OpKind::ConstInt(0),
        };
        let mut transformer = Transformer::new(&config);
        let rewrite = transformer
            ._handle_list_call(
                "list.int_len",
                &op,
                std::slice::from_ref(&l),
                &mut graph,
                "list_int_len",
            )
            .expect("list.int_len must lower");
        let RewriteResult::Replace(ops) = rewrite else {
            panic!("expected Replace");
        };
        assert_eq!(ops.len(), 1);
        match &ops[0].kind {
            OpKind::FieldRead {
                base,
                field,
                ty,
                pure,
            } => {
                assert_eq!(base, &l);
                assert_eq!(field.name, "int_items.len");
                assert_eq!(field.owner_root.as_deref(), Some("W_ListObject"));
                assert!(matches!(ty, ValueType::Int));
                assert!(!pure);
            }
            other => panic!("expected FieldRead, got {other:?}"),
        }
        assert_eq!(ops[0].result, Some(result));
    }

    /// Fallback arm: a `newlist_clear` result with no recoverable list
    /// layout — a generic `OBJECTPTR` (`Ptr(GcStruct("object"))`) or a
    /// missing concretetype — is neither the `"list"` header struct nor an
    /// items array, so `newlist_clear_shape` returns `Fallback` and the arm
    /// emits a conservative `new_array_clear(count, arraydescr)` with a
    /// GC-pointer element (`Ref(None)`) and no array identity.  This input
    /// does not occur in a real translation (the rtyper stamps the list
    /// type on the `_ll_alloc_and_clear` result); it is only reachable from
    /// synthetic graphs that leave the result untyped.  The recovered-shape
    /// arms are covered by
    /// `handle_list_call_newlist_clear_resized_lowers_to_newlist_clear` and
    /// `handle_list_call_newlist_clear_fixed_recovers_int_item_ty`.
    #[test]
    fn handle_list_call_newlist_clear_untyped_result_falls_back_to_ref_array() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("newlist_clear");
        let count = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let result = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let op = SpaceOperation {
            result: Some(result.clone()),
            kind: OpKind::ConstInt(0),
        };
        let mut transformer = Transformer::new(&config);
        let rewrite = transformer
            ._handle_list_call(
                "newlist_clear",
                &op,
                std::slice::from_ref(&count),
                &mut graph,
                "newlist_clear",
            )
            .expect("newlist_clear must lower");
        let RewriteResult::Replace(ops) = rewrite else {
            panic!("expected Replace");
        };
        assert_eq!(ops.len(), 1);
        match &ops[0].kind {
            OpKind::NewArrayClear {
                length,
                item_ty,
                array_type_id,
            } => {
                assert_eq!(length, &count);
                assert!(matches!(item_ty, ValueType::Ref(None)));
                assert_eq!(array_type_id, &None);
            }
            other => panic!("expected NewArrayClear, got {other:?}"),
        }
        assert_eq!(ops[0].result, Some(result));
    }

    /// C1 fork, resized layout: a result typed
    /// `Ptr(GcStruct("list", {length, items: Ptr(GcArray(Signed))}))` — the
    /// resized `ListRepr` shape (`do_resizable_newlist_clear`,
    /// jtransform.py:1938) — lowers to `NewListClear` (the struct-header
    /// compound op) with `item_ty` recovered from the items array element
    /// (`Signed` → `Int`), NOT the pre-fork bare `new_array_clear`.
    #[test]
    fn handle_list_call_newlist_clear_resized_lowers_to_newlist_clear() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            Array, LowLevelType, Ptr, PtrTarget, Struct,
        };
        use crate::translator::rtyper::rtyper::variable_with_lltype;

        // upstream `GcStruct("list", ("length", Signed), ("items",
        // Ptr(GcArray(ITEM))))` with ITEM = Signed (an int-element list).
        let items_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(Array::gc(LowLevelType::Signed)),
        }));
        let list_struct = Struct::gc(
            "list",
            vec![
                ("length".to_string(), LowLevelType::Signed),
                ("items".to_string(), items_ptr),
            ],
        );
        let list_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(list_struct),
        }));

        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("newlist_clear_resized");
        let count = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let result = variable_with_lltype("l", list_ptr);
        let op = SpaceOperation {
            result: Some(result.clone()),
            kind: OpKind::ConstInt(0),
        };
        let mut transformer = Transformer::new(&config);
        let rewrite = transformer
            ._handle_list_call(
                "newlist_clear",
                &op,
                std::slice::from_ref(&count),
                &mut graph,
                "newlist_clear_resized",
            )
            .expect("newlist_clear must lower");
        let RewriteResult::Replace(ops) = rewrite else {
            panic!("expected Replace");
        };
        assert_eq!(ops.len(), 1);
        match &ops[0].kind {
            OpKind::NewListClear {
                length,
                item_ty,
                array_type_id,
            } => {
                assert_eq!(length, &count);
                assert!(
                    matches!(item_ty, ValueType::Int),
                    "int-element list items array must recover ValueType::Int, got {item_ty:?}"
                );
                assert_eq!(array_type_id, &None);
            }
            other => panic!("expected NewListClear, got {other:?}"),
        }
        assert_eq!(ops[0].result, Some(result));
    }

    /// C1 fork, fixed layout: a result typed `Ptr(GcArray(Signed))` — the
    /// fixed `FixedSizeListRepr` shape (`do_fixed_newlist_clear`,
    /// jtransform.py:1858) — lowers to `NewArrayClear` with `item_ty`
    /// recovered from the array element (`Signed` → `Int`), proving the C1
    /// element-type recovery: the pre-fork arm hardcoded `Ref(None)`, which
    /// would GC-trace int slots as pointers.
    #[test]
    fn handle_list_call_newlist_clear_fixed_recovers_int_item_ty() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            Array, LowLevelType, Ptr, PtrTarget,
        };
        use crate::translator::rtyper::rtyper::variable_with_lltype;

        let array_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(Array::gc(LowLevelType::Signed)),
        }));

        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("newlist_clear_fixed");
        let count = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let result = variable_with_lltype("a", array_ptr);
        let op = SpaceOperation {
            result: Some(result.clone()),
            kind: OpKind::ConstInt(0),
        };
        let mut transformer = Transformer::new(&config);
        let rewrite = transformer
            ._handle_list_call(
                "newlist_clear",
                &op,
                std::slice::from_ref(&count),
                &mut graph,
                "newlist_clear_fixed",
            )
            .expect("newlist_clear must lower");
        let RewriteResult::Replace(ops) = rewrite else {
            panic!("expected Replace");
        };
        assert_eq!(ops.len(), 1);
        match &ops[0].kind {
            OpKind::NewArrayClear {
                length,
                item_ty,
                array_type_id,
            } => {
                assert_eq!(length, &count);
                assert!(
                    matches!(item_ty, ValueType::Int),
                    "int-element fixed array must recover ValueType::Int (NOT Ref(None)), \
                     got {item_ty:?}"
                );
                assert_eq!(array_type_id, &None);
            }
            other => panic!("expected NewArrayClear, got {other:?}"),
        }
        assert_eq!(ops[0].result, Some(result));
    }

    /// End-to-end rendezvous: a minted throwaway helper graph is given the
    /// `newlist_clear(count)` oopspec via [`CallControl::mark_oopspec`]
    /// (the same field [`CallControl::get_oopspec`] reads at
    /// `handle_builtin_call`), so a call to it classifies `Builtin` and
    /// dispatches through `_handle_list_call` to `new_array_clear`.  This
    /// locks the T2↔T3 contract: `mark_oopspec(path, "newlist_clear(count)")`
    /// is exactly what T3 must attach to the minted `_ll_alloc_and_clear`
    /// graph for it to lower instead of residualizing.
    #[test]
    fn marked_newlist_clear_oopspec_dispatches_call_to_new_array_clear() {
        let path = crate::parse::CallPath::from_segments(["_ll_alloc_and_clear"]);
        let target = CallTarget::function_path(["_ll_alloc_and_clear"]);

        let mut cc = crate::call::CallControl::new();
        // The spec STRING form the codewriter reader parses: bare name
        // before `(`, so the dispatch keys on `newlist_clear`.
        cc.mark_oopspec(path, "newlist_clear(count)".to_string());

        let mut graph = FunctionGraph::new("caller");
        let count = graph.alloc_value_var();
        let result = graph.alloc_value_var();
        FunctionGraph::set_concretetype_of_inline(&count, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&result, ConcreteType::GcRef);
        let startblock = graph.startblock;
        graph.push_op_var(
            startblock,
            OpKind::Call {
                target,
                args: vec![count.clone()],
                result_ty: ValueType::Ref(None),
            },
            false,
        );
        graph
            .block_mut(startblock)
            .operations
            .last_mut()
            .unwrap()
            .result = Some(result.clone());

        let config = GraphTransformConfig::default();
        let transformed = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .transform(&graph);

        let new_array_clear = transformed
            .graph
            .block(startblock)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                // The GcRef result stamps a generic OBJECTPTR (not a
                // `"list"` struct), so the fork lands on Fallback →
                // NewArrayClear, never NewListClear.
                OpKind::NewArrayClear { length, .. } => Some(length.clone()),
                _ => None,
            })
            .expect("marked newlist_clear must lower to new_array_clear");
        assert_eq!(new_array_clear, count);
    }

    /// T3 end-to-end: the rtyper mints the `_ll_alloc_and_set` family (whose
    /// clear sub-helper is named `_ll_alloc_and_clear` for the resized layout
    /// and `_ll_fixed_alloc_and_clear` for the fixed layout), and `lib.rs`
    /// attaches the `newlist_clear(count)` oopspec under BOTH single-segment
    /// paths.  A `direct_call` to the resized `_ll_alloc_and_clear` funcptr
    /// therefore resolves to that path and lowers to `new_array_clear` — this
    /// locks that the path `lib.rs` registers is exactly the one the minted
    /// graph's name produces.
    #[test]
    fn t3_ll_alloc_and_clear_oopspec_registration_lowers_to_new_array_clear() {
        // Exactly the registration `lib.rs` performs beside the jit.* specs.
        let mut cc = crate::call::CallControl::new();
        cc.mark_oopspec(
            crate::parse::CallPath::from_segments(["_ll_alloc_and_clear"]),
            "newlist_clear(count)".to_string(),
        );

        let mut graph = FunctionGraph::new("caller");
        let count = graph.alloc_value_var();
        let result = graph.alloc_value_var();
        FunctionGraph::set_concretetype_of_inline(&count, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&result, ConcreteType::GcRef);
        let startblock = graph.startblock;
        graph.push_op_var(
            startblock,
            OpKind::Call {
                target: CallTarget::function_path(["_ll_alloc_and_clear"]),
                args: vec![count.clone()],
                result_ty: ValueType::Ref(None),
            },
            false,
        );
        graph
            .block_mut(startblock)
            .operations
            .last_mut()
            .unwrap()
            .result = Some(result.clone());

        let config = GraphTransformConfig::default();
        let transformed = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .transform(&graph);

        let lowered = transformed
            .graph
            .block(startblock)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                // GcRef result → generic OBJECTPTR → Fallback → NewArrayClear.
                OpKind::NewArrayClear { length, .. } => Some(length.clone()),
                _ => None,
            })
            .expect("minted _ll_alloc_and_clear must lower to new_array_clear");
        assert_eq!(lowered, count);
    }

    /// The fixed layout mints its clear sub-helper as `_ll_fixed_alloc_and_clear`
    /// (per `alloc_and_set_sub_names`), so `lib.rs` must register the
    /// `newlist_clear(count)` oopspec on that path too.  A `direct_call` to a
    /// `_ll_fixed_alloc_and_clear` funcptr must lower to `new_array_clear`,
    /// exactly like the resized `_ll_alloc_and_clear` — locking that both
    /// per-layout clear names carry the spec.
    #[test]
    fn t3_ll_fixed_alloc_and_clear_oopspec_registration_lowers_to_new_array_clear() {
        // Both leaf paths `lib.rs` registers beside the jit.* specs.
        let mut cc = crate::call::CallControl::new();
        for clear_name in ["_ll_alloc_and_clear", "_ll_fixed_alloc_and_clear"] {
            cc.mark_oopspec(
                crate::parse::CallPath::from_segments([clear_name]),
                "newlist_clear(count)".to_string(),
            );
        }

        let mut graph = FunctionGraph::new("caller");
        let count = graph.alloc_value_var();
        let result = graph.alloc_value_var();
        FunctionGraph::set_concretetype_of_inline(&count, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&result, ConcreteType::GcRef);
        let startblock = graph.startblock;
        graph.push_op_var(
            startblock,
            OpKind::Call {
                target: CallTarget::function_path(["_ll_fixed_alloc_and_clear"]),
                args: vec![count.clone()],
                result_ty: ValueType::Ref(None),
            },
            false,
        );
        graph
            .block_mut(startblock)
            .operations
            .last_mut()
            .unwrap()
            .result = Some(result.clone());

        let config = GraphTransformConfig::default();
        let transformed = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .transform(&graph);

        let lowered = transformed
            .graph
            .block(startblock)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                // GcRef result → generic OBJECTPTR → Fallback → NewArrayClear.
                OpKind::NewArrayClear { length, .. } => Some(length.clone()),
                _ => None,
            })
            .expect("minted _ll_fixed_alloc_and_clear must lower to new_array_clear");
        assert_eq!(lowered, count);
    }

    /// `list.int_getitem(l, i)` lowers to `getfield_gc_r(l,
    /// int_items.block)` feeding `getarrayitem_gc_i(block, i)`.
    #[test]
    fn handle_list_call_int_getitem_lowers_to_block_plus_getarrayitem() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("list_int_getitem");
        let l = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let index = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let result = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let op = SpaceOperation {
            result: Some(result.clone()),
            kind: OpKind::ConstInt(0),
        };
        let mut transformer = Transformer::new(&config);
        let rewrite = transformer
            ._handle_list_call(
                "list.int_getitem",
                &op,
                &[l.clone(), index.clone()],
                &mut graph,
                "list_int_getitem",
            )
            .expect("list.int_getitem must lower");
        let RewriteResult::Replace(ops) = rewrite else {
            panic!("expected Replace");
        };
        assert_eq!(ops.len(), 2);
        let block = match &ops[0].kind {
            OpKind::FieldRead {
                base, field, ty, ..
            } => {
                assert_eq!(base, &l);
                assert_eq!(field.name, "int_items.block");
                assert_eq!(field.owner_root.as_deref(), Some("W_ListObject"));
                assert!(matches!(ty, ValueType::Ref(None)));
                ops[0].result.clone().expect("block result var")
            }
            other => panic!("expected FieldRead, got {other:?}"),
        };
        match &ops[1].kind {
            OpKind::ArrayRead {
                base,
                index: idx,
                item_ty,
                array_type_id,
                nolength,
                pure,
            } => {
                assert_eq!(base, &block);
                assert_eq!(idx, &index);
                assert!(matches!(item_ty, ValueType::Int));
                assert_eq!(array_type_id, &None);
                assert!(!nolength);
                // `list.int_getitem` is the mutable (non-foldable) read.
                assert!(!pure);
            }
            other => panic!("expected ArrayRead, got {other:?}"),
        }
        assert_eq!(ops[1].result, Some(result));
    }

    /// `list.int_getitem_foldable(l, i)` lowers to `getfield_gc_r(l,
    /// int_items.block)` feeding the foldable `getarrayitem_gc_i_pure(block,
    /// i)` (rlist.py:721-724 `ll_getitem_foldable_nonneg`, oopspec
    /// `list.getitem_foldable`).  The element load is `pure: true`; the
    /// block FieldRead stays `pure: false`.
    #[test]
    fn handle_list_call_int_getitem_foldable_emits_pure_arrayread() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("list_int_getitem_foldable");
        let l = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let index = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let result = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let op = SpaceOperation {
            result: Some(result.clone()),
            kind: OpKind::ConstInt(0),
        };
        let mut transformer = Transformer::new(&config);
        let rewrite = transformer
            ._handle_list_call(
                "list.int_getitem_foldable",
                &op,
                &[l.clone(), index.clone()],
                &mut graph,
                "list_int_getitem_foldable",
            )
            .expect("list.int_getitem_foldable must lower");
        let RewriteResult::Replace(ops) = rewrite else {
            panic!("expected Replace");
        };
        assert_eq!(ops.len(), 2);
        let block = match &ops[0].kind {
            OpKind::FieldRead {
                base,
                field,
                ty,
                pure,
            } => {
                assert_eq!(base, &l);
                assert_eq!(field.name, "int_items.block");
                assert_eq!(field.owner_root.as_deref(), Some("W_ListObject"));
                assert!(matches!(ty, ValueType::Ref(None)));
                // Only the element load is foldable; the block pointer read
                // is not.
                assert!(!pure);
                ops[0].result.clone().expect("block result var")
            }
            other => panic!("expected FieldRead, got {other:?}"),
        };
        match &ops[1].kind {
            OpKind::ArrayRead {
                base,
                index: idx,
                item_ty,
                array_type_id,
                nolength,
                pure,
            } => {
                assert_eq!(base, &block);
                assert_eq!(idx, &index);
                assert!(matches!(item_ty, ValueType::Int));
                assert_eq!(array_type_id, &None);
                assert!(!nolength);
                // The foldable element load — `getarrayitem_gc_i_pure`.
                assert!(pure);
            }
            other => panic!("expected ArrayRead, got {other:?}"),
        }
        assert_eq!(ops[1].result, Some(result));
    }

    /// `list.int_setitem(l, i, v)` lowers to `getfield_gc_r(l,
    /// int_items.block)` feeding `setarrayitem_gc(block, i, v)`.
    #[test]
    fn handle_list_call_int_setitem_lowers_to_block_plus_setarrayitem() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("list_int_setitem");
        let l = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let index = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let value = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let op = SpaceOperation {
            result: None,
            kind: OpKind::ConstInt(0),
        };
        let mut transformer = Transformer::new(&config);
        let rewrite = transformer
            ._handle_list_call(
                "list.int_setitem",
                &op,
                &[l.clone(), index.clone(), value.clone()],
                &mut graph,
                "list_int_setitem",
            )
            .expect("list.int_setitem must lower");
        let RewriteResult::Replace(ops) = rewrite else {
            panic!("expected Replace");
        };
        assert_eq!(ops.len(), 2);
        let block = match &ops[0].kind {
            OpKind::FieldRead { base, field, .. } => {
                assert_eq!(base, &l);
                assert_eq!(field.name, "int_items.block");
                ops[0].result.clone().expect("block result var")
            }
            other => panic!("expected FieldRead, got {other:?}"),
        };
        match &ops[1].kind {
            OpKind::ArrayWrite {
                base,
                index: idx,
                value: written,
                item_ty,
                array_type_id,
                nolength,
            } => {
                assert_eq!(base, &block);
                assert_eq!(idx, &index);
                assert_eq!(written.as_variable(), Some(&value));
                assert!(matches!(item_ty, ValueType::Int));
                assert_eq!(array_type_id, &None);
                assert!(!nolength);
            }
            other => panic!("expected ArrayWrite, got {other:?}"),
        }
        assert_eq!(ops[1].result, None);
    }

    /// `list.int_capacity(l)` lowers to `getfield_gc_r(int_items.block) +
    /// getfield_gc_i(block.capacity)` — capacity is `len(l.items)`
    /// (rlist.py:251), the block's offset-0 length header, not a direct field
    /// of the list.
    #[test]
    fn handle_list_call_int_capacity_lowers_to_block_capacity() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("list_int_capacity");
        let l = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let result = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let op = SpaceOperation {
            result: Some(result.clone()),
            kind: OpKind::ConstInt(0),
        };
        let mut transformer = Transformer::new(&config);
        let rewrite = transformer
            ._handle_list_call(
                "list.int_capacity",
                &op,
                std::slice::from_ref(&l),
                &mut graph,
                "list_int_capacity",
            )
            .expect("list.int_capacity must lower");
        let RewriteResult::Replace(ops) = rewrite else {
            panic!("expected Replace");
        };
        assert_eq!(ops.len(), 2);
        let block = match &ops[0].kind {
            OpKind::FieldRead {
                base, field, ty, ..
            } => {
                assert_eq!(base, &l);
                assert_eq!(field.name, "int_items.block");
                assert!(matches!(ty, ValueType::Ref(_)));
                ops[0].result.clone().expect("block read has a result")
            }
            other => panic!("expected FieldRead, got {other:?}"),
        };
        match &ops[1].kind {
            OpKind::FieldRead {
                base, field, ty, ..
            } => {
                assert_eq!(base, &block);
                assert_eq!(field.name, "capacity");
                assert_eq!(field.owner_root.as_deref(), Some("ItemsBlock"));
                assert!(matches!(ty, ValueType::Int));
            }
            other => panic!("expected FieldRead, got {other:?}"),
        }
        assert_eq!(ops[1].result, Some(result));
    }

    /// `list.int_set_len(l, n)` lowers to a single `setfield_gc_i(l, n,
    /// int_items.len)`.
    #[test]
    fn handle_list_call_int_set_len_lowers_to_len_field_write() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("list_int_set_len");
        let l = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let n = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let op = SpaceOperation {
            result: None,
            kind: OpKind::ConstInt(0),
        };
        let mut transformer = Transformer::new(&config);
        let rewrite = transformer
            ._handle_list_call(
                "list.int_set_len",
                &op,
                &[l.clone(), n.clone()],
                &mut graph,
                "list_int_set_len",
            )
            .expect("list.int_set_len must lower");
        let RewriteResult::Replace(ops) = rewrite else {
            panic!("expected Replace");
        };
        assert_eq!(ops.len(), 1);
        match &ops[0].kind {
            OpKind::FieldWrite {
                base,
                field,
                value,
                ty,
            } => {
                assert_eq!(base, &l);
                assert_eq!(field.name, "int_items.len");
                assert_eq!(value.as_variable(), Some(&n));
                assert!(matches!(ty, ValueType::Int));
            }
            other => panic!("expected FieldWrite, got {other:?}"),
        }
        assert_eq!(ops[0].result, None);
    }

    /// `list.float_len(l)` lowers to a single `getfield_gc_i(l,
    /// float_items.len)` (mirrors `int_len`, addressing the Float block).
    #[test]
    fn handle_list_call_float_len_lowers_to_len_field() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("list_float_len");
        let l = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let result = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let op = SpaceOperation {
            result: Some(result.clone()),
            kind: OpKind::ConstInt(0),
        };
        let mut transformer = Transformer::new(&config);
        let rewrite = transformer
            ._handle_list_call(
                "list.float_len",
                &op,
                std::slice::from_ref(&l),
                &mut graph,
                "list_float_len",
            )
            .expect("list.float_len must lower");
        let RewriteResult::Replace(ops) = rewrite else {
            panic!("expected Replace");
        };
        assert_eq!(ops.len(), 1);
        match &ops[0].kind {
            OpKind::FieldRead {
                base, field, ty, ..
            } => {
                assert_eq!(base, &l);
                assert_eq!(field.name, "float_items.len");
                assert!(matches!(ty, ValueType::Int));
            }
            other => panic!("expected FieldRead, got {other:?}"),
        }
        assert_eq!(ops[0].result, Some(result));
    }

    /// `list.float_setitem(l, i, v)` lowers to `getfield_gc_r(l,
    /// float_items.block)` feeding `setarrayitem_gc_f(block, i, v)` — an
    /// unboxed `f64` store (item type Float, no write barrier).
    #[test]
    fn handle_list_call_float_setitem_lowers_to_setarrayitem_gc_f() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("list_float_setitem");
        let l = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let index = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let value = graph.alloc_value_var_with_type(ConcreteType::Float);
        let op = SpaceOperation {
            result: None,
            kind: OpKind::ConstInt(0),
        };
        let mut transformer = Transformer::new(&config);
        let rewrite = transformer
            ._handle_list_call(
                "list.float_setitem",
                &op,
                &[l.clone(), index.clone(), value.clone()],
                &mut graph,
                "list_float_setitem",
            )
            .expect("list.float_setitem must lower");
        let RewriteResult::Replace(ops) = rewrite else {
            panic!("expected Replace");
        };
        assert_eq!(ops.len(), 2);
        let block = match &ops[0].kind {
            OpKind::FieldRead { base, field, .. } => {
                assert_eq!(base, &l);
                assert_eq!(field.name, "float_items.block");
                ops[0].result.clone().expect("block result var")
            }
            other => panic!("expected FieldRead, got {other:?}"),
        };
        match &ops[1].kind {
            OpKind::ArrayWrite {
                base,
                index: idx,
                value: written,
                item_ty,
                array_type_id,
                nolength,
            } => {
                assert_eq!(base, &block);
                assert_eq!(idx, &index);
                assert_eq!(written.as_variable(), Some(&value));
                assert!(matches!(item_ty, ValueType::Float));
                assert_eq!(array_type_id, &None);
                assert!(!nolength);
            }
            other => panic!("expected ArrayWrite, got {other:?}"),
        }
        assert_eq!(ops[1].result, None);
    }

    /// `list.float_capacity(l)` lowers to `getfield_gc_r(float_items.block) +
    /// getfield_gc_i(block.capacity)` — capacity is `len(l.items)`
    /// (rlist.py:251), the block's offset-0 length header, not a direct field
    /// of the list (mirrors `int_capacity`).
    #[test]
    fn handle_list_call_float_capacity_lowers_to_block_capacity() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("list_float_capacity");
        let l = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let result = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let op = SpaceOperation {
            result: Some(result.clone()),
            kind: OpKind::ConstInt(0),
        };
        let mut transformer = Transformer::new(&config);
        let rewrite = transformer
            ._handle_list_call(
                "list.float_capacity",
                &op,
                std::slice::from_ref(&l),
                &mut graph,
                "list_float_capacity",
            )
            .expect("list.float_capacity must lower");
        let RewriteResult::Replace(ops) = rewrite else {
            panic!("expected Replace");
        };
        assert_eq!(ops.len(), 2);
        let block = match &ops[0].kind {
            OpKind::FieldRead {
                base, field, ty, ..
            } => {
                assert_eq!(base, &l);
                assert_eq!(field.name, "float_items.block");
                assert!(matches!(ty, ValueType::Ref(_)));
                ops[0].result.clone().expect("block read has a result")
            }
            other => panic!("expected FieldRead, got {other:?}"),
        };
        match &ops[1].kind {
            OpKind::FieldRead {
                base, field, ty, ..
            } => {
                assert_eq!(base, &block);
                assert_eq!(field.name, "capacity");
                assert_eq!(field.owner_root.as_deref(), Some("ItemsBlock"));
                assert!(matches!(ty, ValueType::Int));
            }
            other => panic!("expected FieldRead, got {other:?}"),
        }
        assert_eq!(ops[1].result, Some(result));
    }

    /// `list.float_set_len(l, n)` lowers to a single `setfield_gc_i(l, n,
    /// float_items.len)` (mirrors `int_set_len`).
    #[test]
    fn handle_list_call_float_set_len_lowers_to_len_field_write() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("list_float_set_len");
        let l = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let n = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let op = SpaceOperation {
            result: None,
            kind: OpKind::ConstInt(0),
        };
        let mut transformer = Transformer::new(&config);
        let rewrite = transformer
            ._handle_list_call(
                "list.float_set_len",
                &op,
                &[l.clone(), n.clone()],
                &mut graph,
                "list_float_set_len",
            )
            .expect("list.float_set_len must lower");
        let RewriteResult::Replace(ops) = rewrite else {
            panic!("expected Replace");
        };
        assert_eq!(ops.len(), 1);
        match &ops[0].kind {
            OpKind::FieldWrite {
                base,
                field,
                value,
                ty,
            } => {
                assert_eq!(base, &l);
                assert_eq!(field.name, "float_items.len");
                assert_eq!(value.as_variable(), Some(&n));
                assert!(matches!(ty, ValueType::Int));
            }
            other => panic!("expected FieldWrite, got {other:?}"),
        }
        assert_eq!(ops[0].result, None);
    }

    /// An unhandled list oopspec spelling returns `None` so the caller
    /// falls through to the residual-call path.
    #[test]
    fn handle_list_call_unhandled_spelling_returns_none() {
        let config = GraphTransformConfig::default();
        let mut graph = FunctionGraph::new("list_unhandled");
        let l = graph.alloc_value_var_with_type(ConcreteType::GcRef);
        let op = SpaceOperation {
            result: None,
            kind: OpKind::ConstInt(0),
        };
        let mut transformer = Transformer::new(&config);
        assert!(
            transformer
                ._handle_list_call("list.append", &op, &[l], &mut graph, "list_unhandled")
                .is_none()
        );
    }
}
