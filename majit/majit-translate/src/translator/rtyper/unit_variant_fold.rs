//! Pre-jtransform fold of unit-variant `SyntheticTransparentCtor` calls.
//!
//! RPython parity: `rtyper/rpbc.py::SingleFrozenPBCRepr` resolves a
//! frozen PBC constructor that has no arguments (a unit variant
//! `StepResult::Continue`, `JitAction::Continue`, …) into a singleton
//! `Constant(prebuilt_instance_ptr)` before `codewriter/jtransform`
//! ever sees the call.  See also
//! `rclass.InstanceRepr.get_reusable_prebuilt_instance`.
//!
//! Pyre's frontend (`front::mir`) lowers a unit-variant path
//! expression to `OpKind::Call { target: SyntheticTransparentCtor,
//! args: [] }`.  The companion fold inside
//! `translator/rtyper/flowspace_adapter.rs::legacy_const_define_hlvalue`
//! covers graphs that traverse the Match arm of the dual gate, but
//! per-opcode arm body graphs registered via
//! `register_function_graph` take the Skip arm and bypass that fold.
//! The residual `Call` op then survives into jtransform and is emitted
//! as a `residual_call_r/d>r` wrapper, which blocks
//! `production_walker_handles` activation (Task #333).
//!
//! This pass operates directly on `model::FunctionGraph` after
//! `lower_indirect_calls` and before `Transformer::transform`, so it
//! catches both gate arms.  HostObject identity is interned in a
//! process-wide [`UNIT_VARIANT_PREBUILT_INSTANCES`] registry shared
//! with [`legacy_const_define_hlvalue`] so that every graph that
//! references the same unit variant resolves to the *same* prebuilt
//! `HostObject` Arc — mirroring `InstanceRepr.get_reusable_prebuilt_instance`
//! caching on the per-rtyper `instance_reprs` map
//! (`rpython/rtyper/rclass.py:804`, used from
//! `rpython/rtyper/rpbc.py:1026`).  The assembler's `emit_const_r`
//! dedupes the ref-bank constant pool by `obj.identity_id()`, so
//! cross-graph identity sharing collapses the constant pool to a
//! single slot per variant.

use std::sync::{LazyLock, Mutex};

use crate::flowspace::model::HostObject;
use crate::model::{CallTarget, FunctionGraph, OpKind};

/// Process-wide cache of unit-variant prebuilt instance singletons,
/// keyed by qualname.  Mirrors RPython's per-rtyper
/// `instance_reprs[classdef]` cache on top of
/// `InstanceRepr.get_reusable_prebuilt_instance` — every graph
/// referencing `StepResult::Continue` resolves to the same
/// `HostObject` Arc, so downstream `obj.identity_id()` comparisons
/// (assembler ref-bank dedupe, constfold equality, MergePoint
/// greenkey) see a single canonical instance per variant.
///
/// `Vec<(String, HostObject)>` instead of `HashMap` per the
/// project's no-HashMap policy ([[no-hashmap-ever]]).  The variant
/// set is closed and small (~11 entries in
/// [`is_synthetic_unit_variant_path`]), so linear
/// scan is both cheap and PyPy-orthodox.
static UNIT_VARIANT_PREBUILT_INSTANCES: LazyLock<Mutex<Vec<(String, HostObject)>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

/// Find-or-mint the prebuilt singleton instance for an allowlisted
/// unit-variant ctor (`StepResult::Continue`, `LoopResult::Done`, …).
/// Returns the same `HostObject` Arc across all calls and all
/// graphs, mirroring `InstanceRepr.get_reusable_prebuilt_instance`
/// caching on the per-rtyper `instance_reprs` map.  Returns `None`
/// only if `HostObject::Class` cannot produce a prebuilt instance,
/// which by construction never happens for the allowlisted set
/// (every allowlisted path is a unit-variant enum with no fields,
/// so `reusable_prebuilt_instance()` always materialises the
/// `OnceLock` instance — see
/// `majit-translate/src/flowspace/model.rs:394-406`).
pub(crate) fn intern_unit_variant_prebuilt_instance(qualname: &str) -> Option<HostObject> {
    let mut cache = UNIT_VARIANT_PREBUILT_INSTANCES
        .lock()
        .expect("UNIT_VARIANT_PREBUILT_INSTANCES Mutex poisoned");
    if let Some((_, instance)) = cache.iter().find(|(q, _)| q == qualname) {
        return Some(instance.clone());
    }
    let class_obj = HostObject::new_class(qualname, Vec::new());
    let instance = class_obj.reusable_prebuilt_instance()?;
    cache.push((qualname.to_string(), instance.clone()));
    Some(instance)
}

/// Pyre-side `Class::Variant` unit-variant ctors.  These are valid
/// as bare path-expression values; `flowspace_adapter` pre-folds them
/// to `Hlvalue::Constant(ConstValue::HostObject(prebuilt_instance))`
/// before the rtyper sees a call (mirrors PyPy `rtyper` resolving
/// `SomePBC([InstanceDesc(<unit-variant>)])` to a singleton constant
/// before `jtransform`).  Read by [`fold_unit_variant_ctors`] here and
/// by `flowspace_adapter::is_synthetic_unit_variant_call`.
pub(crate) fn is_synthetic_unit_variant_path(segments: &[String]) -> bool {
    let path: Vec<&str> = segments.iter().map(String::as_str).collect();
    matches!(
        path.as_slice(),
        ["LoopResult", "Done"]
            | ["LoopResult", "ContinueRunningNormally"]
            | ["JitAction", "Return"]
            | ["JitAction", "Continue"]
            | ["StepResult", "Continue"]
            | ["CompareOp", "Lt"]
            | ["CompareOp", "Le"]
            | ["CompareOp", "Gt"]
            | ["CompareOp", "Ge"]
            | ["CompareOp", "Eq"]
            | ["CompareOp", "Ne"]
    )
}

/// Rewrite `OpKind::Call { target: SyntheticTransparentCtor, args: [] }`
/// ops whose qualified path matches
/// [`is_synthetic_unit_variant_path`] into
/// `OpKind::ConstRef(prebuilt_instance)`, mirroring
/// `rtyper/rpbc.py::SingleFrozenPBCRepr`.
pub fn fold_unit_variant_ctors(graph: &mut FunctionGraph) {
    for block in graph.blocks.iter_mut() {
        for op in block.operations.iter_mut() {
            let OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { name, owner_path },
                args,
                ..
            } = &op.kind
            else {
                continue;
            };
            if !args.is_empty() {
                continue;
            }
            let mut segments = owner_path.clone();
            segments.push(name.clone());
            if !is_synthetic_unit_variant_path(&segments) {
                continue;
            }
            let qualname = segments.join(".");
            let Some(instance) = intern_unit_variant_prebuilt_instance(&qualname) else {
                continue;
            };
            op.kind = OpKind::ConstRef(instance);
        }
    }
}
