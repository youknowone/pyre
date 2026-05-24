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
/// | `Ref`              | `Instance(SomeInstance{classdef:None,..})` | `model.py:438` `SomeInstance.__init__(classdef, can_be_None=False, flags={})`.  `classdef=None` denotes the abstract `object`-only instance — the legitimate RPython placeholder when the bookkeeper has not narrowed the ClassDef.  Routes through `rclass.py:445-447 SomeInstance.rtyper_makerepr` -> `getinstancerepr(rtyper, None, Gc)` -> `InstanceRepr::new_rootinstance` -> `Ptr(GcStruct(OBJECT))` -> `lowleveltype_to_concrete` GC-arm -> `ConcreteType::GcRef`, matching legacy `resolve_types(Ref) -> GcRef`.  Convergence path: a producer (frontend / annotator with bookkeeper) attaches a richer `SomeInstance(classdef=Some(..))` directly onto `Variable.annotation` once the ClassDef is known (RPython `annrpython.py:289-294 setbinding`). |
/// | `Void`             | `Impossible`           | `model.py:627` -> `SomeValue::Impossible` arm. |
/// | `State`            | `Instance(SomeInstance{classdef:None,..})` | **PRE-EXISTING-ADAPTATION (no upstream)**.  Pyre-only `State` carries the JIT state pointer (a struct pointer to interpreter state).  RPython has no analogue; the `SomeInstance(classdef=None)` projection is a temporary fallback that lets the rtyper proceed without a real bookkeeper-attached pyre `ClassDef`.  Projects to `GcRef` via the same chain as `Ref`. |
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
        ValueType::Ref => {
            // RPython `SomeInstance(classdef=None, can_be_None=False,
            // flags={})` (model.py:438).  Upstream-orthodox abstract
            // instance for cases where the bookkeeper has not yet
            // narrowed the ClassDef.
            //
            // PRE-EXISTING-ADAPTATION (typed-ref-someptr-followup-epic
            // Phase 3, blocked on M2.5g): for typed `&Foo` Rust
            // references the upstream-orthodox lift is
            // `SomePtr(ll_ptrtype)` (`llannotation.py:64-70`), not
            // `SomeInstance(classdef=None)`.  Pyre's
            // `front/ast.rs::classify_fn_arg_ty` collapses `&T`/`*T`
            // into `ValueType::Ref` without a Ptr-carrier variant
            // because the surface DSL has no `lltype.*` HostObject,
            // so this fallback receives both genuine "un-narrowed
            // instance" cases AND typed-ref cases.  Producers that
            // already know the operand is a typed pointer should call
            // `AnnotationState::set_some(vid, SomeValue::Ptr(
            // SomePtr::new(t)))` directly to bypass this fallback —
            // the downstream `cast_ptr_to_int` typer's late
            // InstanceRepr→PtrRepr swap (`rbuiltin.rs:2174-2194`)
            // exists specifically because no such producer is wired
            // for typed `&Foo` yet.  Convergence: M2.5g registry
            // walker lands → frontend lifts typed `&Foo` directly to
            // `SomeValue::Ptr` at `value_type_for_type` time.
            Some(SomeValue::Instance(SomeInstance::new(
                None,
                false,
                BTreeMap::new(),
            )))
        }
        ValueType::State => {
            // PRE-EXISTING-ADAPTATION (no upstream).  Pyre's `State`
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

/// Reduce a `SomeValue` lattice node to its `ValueType` discriminator.
/// Inverse of [`valuetype_to_someshell`].
///
/// RPython parity: `getkind` family in `rpython/rtyper/lltypesystem/lltype.py`
/// reduces lltypes to backend kinds; the analogue here reduces
/// annotation-stage `SomeValue` to pyre's flat `ValueType` enum used by
/// downstream `jit_codewriter` consumers that haven't been ported to
/// `SomeValue` directly.
pub fn somevalue_to_valuetype(s: &SomeValue) -> ValueType {
    match s {
        SomeValue::Integer(_) => ValueType::Int,
        SomeValue::Bool(_) => ValueType::Bool,
        SomeValue::Float(_) | SomeValue::SingleFloat(_) | SomeValue::LongFloat(_) => {
            ValueType::Float
        }
        SomeValue::Instance(_) | SomeValue::Ptr(_) | SomeValue::PBC(_) => ValueType::Ref,
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
        _ => ValueType::Ref,
    }
}
