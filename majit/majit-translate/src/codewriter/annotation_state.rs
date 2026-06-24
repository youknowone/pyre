//! `ValueType` ↔ `SomeValue` shell projection helpers.
//!
//! RPython's `RPythonAnnotator.complete()` attaches a `SomeValue`
//! directly to each `Variable.annotation` slot on the flowgraph
//! (`rpython/annotator/annrpython.py:54-66`,
//! `rpython/flowspace/model.py: Variable.annotation`).  Pyre writes the
//! same lattice node into `Variable.annotation` via
//! [`crate::translator::rtyper::legacy_annotator::setbinding`], which
//! routes through [`valuetype_to_someshell`] to build the shell.
//!
//! `ValueType::Unknown` returns `None` from [`valuetype_to_someshell`]
//! — annotation gaps surface fail-loud at the rtyper's `bindingrepr`
//! instead of being bridged to a fabricated GC reference.

use std::collections::BTreeMap;

use crate::annotator::model::{SomeFloat, SomeInstance, SomeInteger, SomeValue};
use crate::model::ValueType;

/// RPython `SomeValue` lattice projection of the legacy `ValueType`.
///
/// `RPythonTyper.bindingrepr` (`rtyper.rs:961`) dispatches purely on the
/// `SomeValue` shape via `rtyper_makekey` / `rtyper_makerepr`.  `Int`
/// and `Float` resolve cleanly through `rint::IntegerRepr` /
/// `rfloat::FloatRepr`.
///
/// Mapping:
///
/// | legacy `ValueType` | `SomeValue` shell      | RPython source / status |
/// |--------------------|------------------------|--------------------------|
/// | `Int`              | `Integer(SomeInteger)` | `model.py:206-224` -> `SomeValue::Integer` arm. |
/// | `Float`            | `Float(SomeFloat)`     | `model.py:164-183` -> `SomeValue::Float` arm. |
/// | `Ref(_)`           | `Instance(SomeInstance{classdef:None,..})` | `model.py:438`.  Typed pointers should lift to `SomePtr(ll_ptrtype)` (`llannotation.py:64-70`), but the correct Ptr must come from the producer writing `Variable.annotation` directly — not from a process-global root-string lookup.  This fallback projection keeps all Ref variants classdef-less → `GcRef` via `rclass.py:445-447`. |
/// | `Void`             | `Impossible`           | `model.py:627` -> `SomeValue::Impossible` arm. |
/// | `State`            | `Instance(SomeInstance{classdef:None,..})` | **TODO: no upstream equivalent**.  Pyre-only `State` carries the JIT state pointer (a struct pointer to interpreter state).  RPython has no analogue; the `SomeInstance(classdef=None)` projection is a temporary fallback that lets the rtyper proceed without a real bookkeeper-attached pyre `ClassDef`.  Projects to `GcRef` via the same chain as `Ref`. |
/// | `Unknown`          | `None`                 | **Fail-loud — annotation gap with no annotation-stage shell.**  RPython's annotator never produces an unknown lattice node — every Variable is annotated with a definite `SomeValue`, and unreachable code stays at `SomeImpossible`.  Pyre's `Unknown` is a coverage gap (annotator did not narrow / producer did not call `set_some`).  Returning `None` leaves `Variable.annotation` empty so `bindingrepr` panics with `KeyError: no binding for arg` (`annotator/annrpython.rs:418`) on the first attempt to lower the affected `Variable`, surfacing the producer-side gap rather than silently bridging it to `GcRef` via a fabricated `SomeInstance(None)` shell — that bridging conflated an *annotation-stage* lattice node with the **legacy** `resolve_types(Unknown) -> ConcreteType::Unknown -> GcRef` resolver-stage backfill. |
///
/// Returns `None` only for `ValueType::Unknown`; every other variant
/// projects to a definite `SomeValue` shell.
pub fn valuetype_to_someshell(vt: &ValueType) -> Option<SomeValue> {
    match vt {
        // `Int` shells to `SomeInteger { unsigned: false }` (default);
        // `Unsigned` shells to `SomeInteger { unsigned: true }` so the
        // rtyper picks `IntegerRepr.lowleveltype = Unsigned`
        // (`rint.py:_init_repr`).  `getkind(Unsigned) == 'int'` so the
        // codewriter / regalloc share register classes via Int|Unsigned
        // arms downstream.
        ValueType::Int => Some(SomeValue::Integer(SomeInteger::default())),
        ValueType::Unsigned => Some(SomeValue::Integer(SomeInteger::new(false, true))),
        // RPython `SomeBool` (`annotator/model.py:185-198`) is a
        // distinct lattice node from `SomeInteger`; the rtyper picks
        // `BoolRepr` (`rmodel.rs::BoolRepr`) which lowers to LL `Bool`
        // (integer-compatible).  Until a richer Bool annotation lands
        // (with truthy_value tracking per `model.py:188-194`), shape it
        // as `SomeBool::default` matching `SomeBool()` upstream.
        ValueType::Bool => Some(SomeValue::Bool(crate::annotator::model::SomeBool::default())),
        ValueType::Float => Some(SomeValue::Float(SomeFloat::default())),
        ValueType::Ref(_) => {
            // RPython typed pointers lift to `SomePtr(ll_ptrtype)`
            // (`llannotation.py:64-70`), but the correct Ptr must come
            // from the producer (annotator with bookkeeper / host-class
            // registry) writing `Variable.annotation` directly — not
            // from a process-global root-string lookup here.  A global
            // bare-name index conflates cross-module same-name structs
            // and bypasses RPython's object-identity-based lltype cache.
            // The MIR front-end does not attach a per-`&Foo`-input
            // lltype `Ptr` to `Variable.annotation` (the faithful
            // counterpart of `lltype.py:1513-1518
            // _ptrEntry.compute_annotation`), so every `Ref` input
            // shells to the classdef-less `SomeInstance` here rather
            // than fabricating a `Ptr` from a bare name.  Routing a
            // host-registry lookup back into *this* shell projector
            // would be the rejected bare-name path.
            Some(ref_fallback_instance())
        }
        ValueType::State => {
            // TODO: no upstream equivalent.  Pyre's `State`
            // carries the JIT state pointer; RPython has no analogue.
            // `SomeInstance(classdef=None)` is a temporary fallback
            // that lets the rtyper resolve to `GcRef` without a real
            // bookkeeper-attached pyre `ClassDef`.
            Some(SomeValue::Instance(SomeInstance::new(
                None,
                false,
                BTreeMap::new(),
            )))
        }
        ValueType::Unknown => {
            // Fail-loud — annotation gap, NOT annotation-stage parity.
            // RPython's annotator never produces an unknown lattice
            // node; pyre's `Unknown` is a coverage gap.  Returning
            // `None` leaves `Variable.annotation` empty so the rtyper
            // panics at `bindingrepr` (annotator/annrpython.rs:418
            // "KeyError: no binding for arg") on the first attempt to
            // lower an Unknown Variable.  This surfaces the producer-
            // side gap rather than silently bridging it to `GcRef` via
            // a fabricated `SomeInstance(None)` shell — that bridging
            // conflated the annotation-stage lattice node with the
            // **legacy** resolver-stage backfill
            // (`resolve_types(Unknown) -> ConcreteType::Unknown ->
            // GcRef`).  Convergence path: precise producer-side
            // `set_some` for every `Variable`.
            None
        }
        ValueType::Void => Some(SomeValue::Impossible),
    }
}

fn ref_fallback_instance() -> SomeValue {
    SomeValue::Instance(SomeInstance::new(None, false, BTreeMap::new()))
}

/// Reduce a `SomeValue` lattice node to its `ValueType` discriminator.
/// Inverse of [`valuetype_to_someshell`].
///
/// RPython parity: `getkind` family in `rpython/rtyper/lltypesystem/lltype.py`
/// reduces lltypes to backend kinds; the analogue here reduces
/// annotation-stage `SomeValue` to pyre's flat `ValueType` enum used by
/// downstream `codewriter` consumers that haven't been ported to
/// `SomeValue` directly.
pub fn somevalue_to_valuetype(s: &SomeValue) -> ValueType {
    match s {
        SomeValue::Integer(_) => ValueType::Int,
        SomeValue::Bool(_) => ValueType::Bool,
        SomeValue::Float(_) | SomeValue::LongFloat(_) => ValueType::Float,
        // `getkind(SingleFloat) == 'int'` (history.py:53): singlefloats
        // are stored in an int register, not the float bank.
        SomeValue::SingleFloat(_) => ValueType::Int,
        SomeValue::Instance(_) | SomeValue::Ptr(_) | SomeValue::PBC(_) => ValueType::Ref(None),
        // `SomeImpossibleValue` represents unreachable code (`model.py:627`),
        // which projects to `ValueType::Void` in pyre's flat enum just
        // like upstream `lltype.Void`.
        SomeValue::Impossible => ValueType::Void,
        // Other variants (String / List / Tuple / Dict / Iterator /
        // Exception / None_ / Property / InteriorPtr / LLADTMeth /
        // Builtin / BuiltinMethod / WeakRef / TypeOf / ByteArray /
        // Char / UnicodeCodePoint / UnicodeString / Type / Object) have
        // no direct pyre `ValueType` mapping. Consumers that need the
        // precise lattice node should read the `Variable.annotation`
        // shell directly; this reduced projection falls back to `Ref`
        // so GC-pointer lowering applies.
        _ => ValueType::Ref(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_ref_uses_instance_fallback() {
        let shell = valuetype_to_someshell(&ValueType::Ref(Some("SomeStruct".to_string())))
            .expect("Ref projects");
        match shell {
            SomeValue::Instance(inst) => {
                assert!(inst.classdef.is_none());
                assert!(!inst.can_be_none);
                assert!(inst.flags.is_empty());
            }
            other => panic!("typed Ref must use fallback Instance, got {other:?}"),
        }
    }
}
