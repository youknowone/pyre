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
//! graphs registered directly through `register_function_graph` can
//! take the Skip arm and bypass that fold.
//! The residual `Call` op then survives into jtransform and is emitted
//! as a `residual_call_r/d>r` wrapper, which blocks the JitCode walker.
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

use parking_lot::Mutex;
use std::sync::LazyLock;

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
/// `majit-translate/src/flowspace/model.rs`).
pub(crate) fn intern_unit_variant_prebuilt_instance(qualname: &str) -> Option<HostObject> {
    let mut cache = UNIT_VARIANT_PREBUILT_INSTANCES.lock();
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
///
/// An LLBC-derived graph spells the ctor with its full module path and
/// generic instantiation (`resolve_aggregate_adt` pushes the
/// per-instantiation leaf, e.g. `pyre_interpreter::pyopcode::
/// StepResult<*mut PyObject>::Continue`), so the owner segment is
/// compared instantiation-stripped, the same way `result_ctor_kind`
/// strips `Result<T, E>` before comparing.
pub(crate) fn is_synthetic_unit_variant_path(segments: &[String]) -> bool {
    let [head @ .., owner, name] = segments else {
        return false;
    };
    let owner_base = owner.split_once('<').map_or(owner.as_str(), |(b, _)| b);
    match (owner_base, name.as_str()) {
        ("LoopResult", "Done" | "ContinueRunningNormally")
        | ("JitAction", "Return" | "Continue")
        | ("CompareOp", "Lt" | "Le" | "Gt" | "Ge" | "Eq" | "Ne") => head.is_empty(),
        ("StepResult", "Continue") => head.is_empty() || head == ["pyre_interpreter", "pyopcode"],
        _ => false,
    }
}

/// Whether `name` is a shaped positional aggregate carrying no items:
/// `Tuple<>` or `Array<T;0>`.
///
/// The shape suffix is the aggregate's low-level identity, so an empty one is
/// decidable from the name alone. `Tuple` and `Array` without a suffix are the
/// bare roots and are not this: the bare `Tuple` is the unit handled above, and
/// a bare `Array` carries no length to be zero.
fn is_zero_length_shaped_aggregate(name: &str) -> bool {
    if majit_ir::descr::is_shaped_tuple_name(name) {
        return name
            .strip_prefix("Tuple<")
            .and_then(|rest| rest.strip_suffix('>'))
            .is_some_and(|args| args.trim().is_empty());
    }
    majit_ir::descr::is_shaped_array_name(name)
        && name
            .rsplit_once(';')
            .and_then(|(_, len)| len.strip_suffix('>'))
            .is_some_and(|len| len.trim() == "0")
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
                target:
                    CallTarget::SyntheticTransparentCtor {
                        name, owner_path, ..
                    },
                args,
                ..
            } = &op.kind
            else {
                continue;
            };
            if !args.is_empty() {
                continue;
            }
            // A 0-arg `Tuple` transparent ctor is the Rust unit `()` value
            // — a ZST carrying no runtime data.  Lower it to the pure null
            // ref so a void function's dead `()` producer is a pure op that
            // `prune_dead_phis` can DCE, instead of a non-pure ctor `Call`
            // that survives into regalloc and collides a register with a
            // live parameter.
            if owner_path.is_empty() && name == "Tuple" {
                op.kind = OpKind::ConstRefNull;
                continue;
            }
            // A zero-length shaped aggregate — `Tuple<>`, `Array<T;0>`, the
            // empty argument slice `&[]` — carries no runtime data either, but
            // unlike the unit above its value IS read: it flows on as a call
            // argument.  Upstream answers both halves the same way and neither
            // one allocates: `rtuple.TUPLE_TYPE` returns `Void` for an empty
            // field list before any `GcStruct` exists, `TupleRepr.newtuple`
            // returns `inputconst(Void, ())` instead of emitting a `malloc`,
            // and `TupleRepr.instantiate` hands back the prebuilt
            // `dum_empty_tuple` PBC.  A zero-length `FixedSizeArray` is
            // excluded more strongly still: it inherits `Struct._gckind =
            // 'raw'`, so `lltype.malloc(flavor='gc')` refuses it outright.
            //
            // So the value is a prebuilt singleton, not an allocation. It must
            // also be non-null, which is why this arm interns an instance where
            // the unit above emits the null ref: a caller passes it on, and the
            // walker refuses a null ref argument to a may-force call.
            //
            // Leaving it as an allocation is what
            // `register_synthetic_positional_metadata` registers with zero rows
            // and no collector-issued type id, which the walker rejects with
            // `UnregisteredNewGcType` after the descent has already run.
            if owner_path.is_empty()
                && is_zero_length_shaped_aggregate(name)
                && let Some(instance) = intern_unit_variant_prebuilt_instance(name)
            {
                op.kind = OpKind::ConstRef(instance);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_length_shapes_are_recognised() {
        assert!(is_zero_length_shaped_aggregate("Tuple<>"));
        assert!(is_zero_length_shaped_aggregate("Array<*mut PyObject;0>"));
        assert!(is_zero_length_shaped_aggregate("Array<u8;0>"));
    }

    /// The bare roots are not shaped, and a shape that carries items is not
    /// empty — both keep the allocation.
    #[test]
    fn bare_roots_and_non_empty_shapes_are_rejected() {
        assert!(!is_zero_length_shaped_aggregate("Tuple"));
        assert!(!is_zero_length_shaped_aggregate("Array"));
        assert!(!is_zero_length_shaped_aggregate("Tuple<*mut PyObject>"));
        assert!(!is_zero_length_shaped_aggregate("Array<*mut PyObject;1>"));
        assert!(!is_zero_length_shaped_aggregate("Array<u8;10>"));
    }

    /// A length that merely ENDS in `0` is not length zero.
    #[test]
    fn a_length_ending_in_zero_is_not_zero() {
        assert!(!is_zero_length_shaped_aggregate("Array<u8;20>"));
        assert!(!is_zero_length_shaped_aggregate("Array<u8;100>"));
    }

    /// The qualified, generic-instantiated spelling an LLBC graph carries
    /// must match alongside the bare two-segment form; unrelated paths
    /// and other variants of the same enum must not.
    #[test]
    fn qualified_instantiated_unit_variant_paths_match() {
        let q = |s: &[&str]| s.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(is_synthetic_unit_variant_path(&q(&[
            "StepResult",
            "Continue"
        ])));
        assert!(is_synthetic_unit_variant_path(&q(&[
            "pyre_interpreter",
            "pyopcode",
            "StepResult<*mut PyObject>",
            "Continue",
        ])));
        assert!(!is_synthetic_unit_variant_path(&q(&[
            "pyre_interpreter",
            "pyopcode",
            "StepResult<*mut PyObject>",
            "Yield",
        ])));
        assert!(!is_synthetic_unit_variant_path(&q(&[
            "other_crate",
            "StepResult",
            "Continue",
        ])));
        assert!(!is_synthetic_unit_variant_path(&q(&[
            "JitAction",
            "Continue",
            "Extra"
        ])));
        assert!(!is_synthetic_unit_variant_path(&q(&["Continue"])));
    }

    /// One prebuilt instance per shape, shared across graphs, as
    /// `InstanceRepr.get_reusable_prebuilt_instance` caches per classdef.
    #[test]
    fn one_prebuilt_instance_per_shape() {
        let a = intern_unit_variant_prebuilt_instance("Array<*mut PyObject;0>").unwrap();
        let b = intern_unit_variant_prebuilt_instance("Array<*mut PyObject;0>").unwrap();
        let other = intern_unit_variant_prebuilt_instance("Array<u8;0>").unwrap();
        assert_eq!(a.identity_id(), b.identity_id());
        assert_ne!(a.identity_id(), other.identity_id());
    }
}
