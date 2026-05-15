//! Narrow semantic graph scaffold for the future graph-based translator.
//!
//! This is intentionally much smaller than a full Rust compiler IR.  It exists
//! to model only the semantics needed by majit's translation/codewriter layer.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;

use crate::flowspace::model::ConstValue;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ValueId(pub usize);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueType {
    Int,
    /// `lltype.Unsigned` — register class is `'int'` per
    /// `getkind(Unsigned) == 'int'`, distinct from `Signed` only at
    /// the rtyper dispatch level (`rbool.py:78 uint_is_true`,
    /// `rint.py` cast family).  Downstream consumers that do not
    /// distinguish signedness (regalloc, jit_codewriter,
    /// valuetype_to_someshell) treat `Unsigned` identically to `Int`
    /// via `Int | Unsigned` arms.
    ///
    /// SCAFFOLDED, NOT YET REACHED FROM PYRE SOURCE.  The producer
    /// (`front/ast.rs:classify_fn_arg_ty`) currently folds Rust
    /// `u8`/`u16`/`u32`/`u64`/`usize` into `Int` because flipping
    /// them onto this arm cascades into pyre-source analysis
    /// regressions (`annotator/unaryop.rs:445` getattr,
    /// `assembler.rs:581` kind-mismatch in `setinteriorfield_gc_r`).
    /// The `cast_op_name` Unsigned arms (`cast_uint_to_*`,
    /// `cast_*_to_uint`, `uint_is_true`) and the matching rtyper
    /// handlers (`rtype_cast_uint_to_float` /
    /// `rtype_cast_float_to_uint` / `rtype_uint_is_true`) are wired
    /// for the eventual flip — see `front/ast.rs:classify_fn_arg_ty`
    /// TODO(unsigned-producer-flip) for the convergence path.
    Unsigned,
    /// RPython `SomeBool` (`annotator/model.py:185-198`): a Python `bool`
    /// at the annotator level.  Distinct from `Int` (RPython `SomeInteger`,
    /// `:200-264`) because PyPy's flowspace `UNARY_NOT` (`flowcontext.py:531-538`,
    /// `op.bool` then `guessbool` fork) and `UNARY_INVERT` (`:188-191`,
    /// `op.invert`) dispatch on which one is on the stack:
    /// `not` for booleans, `~` for integers.
    ///
    /// Every comparison opname (`lt`/`le`/`eq`/`ne`/`gt`/`ge`,
    /// `operation.py:505-510 add_operator(.., dispatch=2, pure=True)`),
    /// `is`/`is not`, the unary `bool` op, and Rust `bool`-typed locals
    /// produce `Bool`.
    ///
    /// `Lit::Bool` and the `UNARY_NOT` arms in `front/ast.rs` emit
    /// the dedicated `OpKind::ConstBool(bool)` variant; the rtyper
    /// adapter lifts it to `Constant(True/False, lltype.Bool)` so the
    /// annotator selects `SomeBool` and the rtyper picks `BoolRepr`.
    /// The codewriter collapses storage to the int kind via
    /// `getkind(Bool) == 'int'`
    /// (`rpython/jit/codewriter/flatten.py:getkind`); the assembler
    /// emits `int_copy/i>i` for both `ConstInt` and `ConstBool`.
    ///
    /// Pyre's annotator-side `valuetype_to_someshell`
    /// (`annotation_state.rs::valuetype_to_someshell`) lifts `Bool`
    /// to `SomeBool(unspecified)`.  The rtyper has `BoolRepr`
    /// available (`rmodel.rs::BoolRepr`); the high-level dispatch in
    /// `RPythonTyper::translate_op` continues to share rint paths for
    /// arithmetic operations because RPython's `BoolRepr.lowleveltype
    /// = Bool` is integer-compatible at the LL level (everything
    /// except logical-vs-bitwise distinguishes).
    Bool,
    Ref,
    Float,
    Void,
    State,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnknownKind {
    /// `syn::Stmt::Macro` — any macro invocation at statement position
    /// (`panic!(...)`, `debug_assert!(...)`, `println!(...)`, etc).
    /// Always skipped in `lower_stmt` (no flow-graph analogue — RPython
    /// source has no macros).
    MacroStmt,
    /// `syn::Expr::Lit` with a kind pyre cannot yet model.  The `variant`
    /// tag names the specific syn literal kind (`Str`, `Float`,
    /// `ByteStr`, `Verbatim`) so downstream diagnostics and
    /// `MAJIT_UNKNOWN_DUMP` logs show the exact failure category
    /// without re-walking the syn AST.
    UnsupportedLiteral { variant: UnsupportedLiteralKind },
    /// `syn::Expr::*` variants pyre cannot yet lower.  Tag names the
    /// specific expression kind so diagnostics and the `Unknown`
    /// markers left in SSA graphs identify the remaining port gap.
    UnsupportedExpr { variant: UnsupportedExprKind },
}

/// Reason an `syn::Lit::*` could not be lowered.  Parity with RPython
/// `flowspace/flowcontext.py` — unsupported literals raise
/// `FlowingError` with a kind-specific message; pyre's analogue
/// records the kind on the Unknown marker op so the abort path is
/// traceable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnsupportedLiteralKind {
    Str,
    Float,
    ByteStr,
    Verbatim,
    Other,
}

/// Reason an `syn::Expr::*` could not be lowered.  Same role as
/// `UnsupportedLiteralKind` — tags the specific `FlowingError`-
/// equivalent abort cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnsupportedExprKind {
    Array,
    Async,
    Await,
    Const,
    ForLoop,
    Group,
    Infer,
    Let,
    Macro,
    Range,
    RawAddr,
    Repeat,
    Struct,
    Tuple,
    TryBlock,
    Verbatim,
    Yield,
    /// Rust `!x` whose operand has no statically-known bool/int type.
    /// RPython distinguishes UNARY_NOT (`flowcontext.py:531-538`) from
    /// UNARY_INVERT (`flowcontext.py:188-191`) at the bytecode token,
    /// so an Unknown operand cannot pick the parity-correct flowspace
    /// op without guessing.
    UnaryNotUnknownOperand,
    /// Multi-segment `Expr::Path` (e.g. `Into::into`, `Foo::bar`) used
    /// as a value rather than as a call target.  RPython has no analogue
    /// — `flowcontext.py:LOAD_GLOBAL` lifts a name lookup to a constant
    /// (registered `FuncDesc` / class) or raises `FlowingError`; pyre
    /// has no `Const(PBCRef)` IR variant for a bare path-as-value, so
    /// classifying these explicitly as `Abort` keeps the producer site
    /// distinct from the "cross-block local was not threaded" category
    /// (the prior fall-through emitted naked `OpKind::Input { name:
    /// "Foo::bar" }` and tripped `adapter cross-block body Input`).
    PathConstantRef,
    /// Rust `|args| body` closure expression used as a value (closure
    /// capture / function-pointer argument).  RPython has no closure
    /// model in flowspace — `flowcontext.py:1235 MAKE_FUNCTION` raises
    /// `FlowingError` for nested-fn / lambda surfaces (only top-level
    /// `FunctionGraph`s round-trip through the bookkeeper).  Pyre's
    /// `syn::Expr::Closure` arm produces an `Unknown` marker for the
    /// closure value rather than walking the body — inlining it would
    /// be a NEW-DEVIATION (treats the closure as a synchronous block).
    /// Classified separately from `OtherExpr` so the producer site is
    /// distinguishable from the catch-all in dual-gate diagnostics.
    Closure,
    OtherExpr,
}

/// RPython `rpython/rtyper/rclass.py:57-60` — `IR_IMMUTABLE` / `IR_IMMUTABLE_ARRAY`
/// / `IR_QUASIIMMUTABLE` / `IR_QUASIIMMUTABLE_ARRAY`.  Parsed from
/// `_immutable_fields_` string literals (`rclass.py:644-678 _parse_field_list`):
///
/// * `"field"`       → `Immutable`
/// * `"field?"`      → `QuasiImmutable`
/// * `"field[*]"`    → `ImmutableArray`
/// * `"field?[*]"`   → `QuasiImmutableArray`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImmutableRank {
    Immutable,
    QuasiImmutable,
    ImmutableArray,
    QuasiImmutableArray,
}

impl ImmutableRank {
    /// Parse a single RPython-style `_immutable_fields_` entry.  Returns the
    /// bare field name and its rank.  Suffix precedence matches
    /// `rclass.py:649-661`: `?[*]` → `[*]` → `?` → plain.
    pub fn parse(entry: &str) -> (String, Self) {
        if let Some(stripped) = entry.strip_suffix("?[*]") {
            (stripped.to_string(), Self::QuasiImmutableArray)
        } else if let Some(stripped) = entry.strip_suffix("[*]") {
            (stripped.to_string(), Self::ImmutableArray)
        } else if let Some(stripped) = entry.strip_suffix('?') {
            (stripped.to_string(), Self::QuasiImmutable)
        } else {
            (entry.to_string(), Self::Immutable)
        }
    }

    /// RPython `ImmutableRanking.pure` flag — `rclass.py:33-37`.  True for
    /// `IR_IMMUTABLE` / `IR_IMMUTABLE_ARRAY`; false for the quasi variants
    /// (they pin via guard, not via pure flag).
    pub fn is_immutable(self) -> bool {
        matches!(self, Self::Immutable | Self::ImmutableArray)
    }

    /// True for `IR_QUASIIMMUTABLE` / `IR_QUASIIMMUTABLE_ARRAY` —
    /// `jtransform.py:895 immut in (IR_QUASIIMMUTABLE, IR_QUASIIMMUTABLE_ARRAY)`.
    pub fn is_quasi_immutable(self) -> bool {
        matches!(self, Self::QuasiImmutable | Self::QuasiImmutableArray)
    }

    /// True for `IR_IMMUTABLE_ARRAY` / `IR_QUASIIMMUTABLE_ARRAY` —
    /// `rclass.py:670 rank in (IR_QUASIIMMUTABLE_ARRAY, IR_IMMUTABLE_ARRAY)`.
    pub fn is_array(self) -> bool {
        matches!(self, Self::ImmutableArray | Self::QuasiImmutableArray)
    }
}

impl Default for ImmutableRank {
    fn default() -> Self {
        Self::Immutable
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CallTarget {
    Method {
        name: String,
        receiver_root: Option<String>,
        /// Plan `annotator-monomorphization-tier1` phase M2.b — transient
        /// transport only for the concrete receiver `ClassDefKey` that
        /// the annotator's `lookup_filter` / `MethodDesc.selfclassdef`
        /// picks per call site. **Never source of truth**: authority is
        /// `MethodDesc(originclassdef/selfclassdef)` +
        /// `SomeInstance.classdef`. This field is `None` everywhere
        /// except in the narrow window between the annotator pass and
        /// the call-resolution fold; serde + stable caches must never
        /// observe `Some(_)` (enforced by `#[serde(skip)]` and the fold
        /// pass that collapses `Some` targets to a concrete `CallPath`
        /// before serialization).
        #[serde(skip, default)]
        classdef_hint: Option<crate::annotator::description::ClassDefKey>,
    },
    FunctionPath {
        segments: Vec<String>,
    },
    /// Rust frontend adaptation for constructors that RPython's rtyper erases
    /// before jtransform. This variant must only be produced after frontend
    /// resolution proves the call is not a user-defined function.
    SyntheticTransparentCtor {
        name: String,
    },
    /// RPython: `indirect_call` opname. Receiver's static type is a
    /// `dyn Trait` (Rust fat pointer); at JIT time the actual callee
    /// is resolved via vtable.  `trait_root` + `method_name` together
    /// key the candidate family in `CallControl.trait_method_impls`.
    Indirect {
        trait_root: String,
        method_name: String,
    },
    UnsupportedExpr,
}

impl CallTarget {
    pub fn method(name: impl Into<String>, receiver_root: Option<String>) -> Self {
        Self::Method {
            name: name.into(),
            receiver_root,
            classdef_hint: None,
        }
    }

    /// Plan M2.b constructor — attach the annotator's concrete-receiver
    /// classdef decision to an already-built `CallTarget::Method`. Used
    /// only by the fold pass that consumes MethodDesc-keyed decisions;
    /// every other caller should go through [`Self::method`] to keep
    /// `classdef_hint` at `None`.
    pub fn method_with_classdef_hint(
        name: impl Into<String>,
        receiver_root: Option<String>,
        classdef_hint: crate::annotator::description::ClassDefKey,
    ) -> Self {
        Self::Method {
            name: name.into(),
            receiver_root,
            classdef_hint: Some(classdef_hint),
        }
    }

    /// Plan M2.b accessor — read the classdef hint if present. Callers
    /// must treat `None` as "annotator did not supply a decision";
    /// falling back to `receiver_root` string is the pre-M3 heuristic
    /// path and should be retired per plan phase M4.
    pub fn classdef_hint(&self) -> Option<crate::annotator::description::ClassDefKey> {
        match self {
            CallTarget::Method { classdef_hint, .. } => *classdef_hint,
            _ => None,
        }
    }

    pub fn function_path<I, S>(segments: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::FunctionPath {
            segments: segments.into_iter().map(Into::into).collect(),
        }
    }

    pub fn indirect(trait_root: impl Into<String>, method_name: impl Into<String>) -> Self {
        Self::Indirect {
            trait_root: trait_root.into(),
            method_name: method_name.into(),
        }
    }

    pub fn synthetic_transparent_ctor(name: impl Into<String>) -> Self {
        Self::SyntheticTransparentCtor { name: name.into() }
    }

    pub fn receiver_root(&self) -> Option<&str> {
        match self {
            CallTarget::Method { receiver_root, .. } => receiver_root.as_deref(),
            _ => None,
        }
    }

    pub fn path_segments(&self) -> Option<Vec<&str>> {
        match self {
            CallTarget::Method { name, .. } => Some(vec![name.as_str()]),
            CallTarget::FunctionPath { segments } => {
                Some(segments.iter().map(String::as_str).collect())
            }
            CallTarget::SyntheticTransparentCtor { name } => Some(vec![name.as_str()]),
            CallTarget::Indirect { method_name, .. } => Some(vec![method_name.as_str()]),
            CallTarget::UnsupportedExpr => None,
        }
    }
}

impl fmt::Display for CallTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CallTarget::Method {
                name,
                receiver_root: Some(receiver_root),
                ..
            } => write!(f, "{receiver_root}.{name}"),
            CallTarget::Method {
                name,
                receiver_root: None,
                ..
            } => f.write_str(name),
            CallTarget::FunctionPath { segments } => f.write_str(&segments.join("::")),
            CallTarget::SyntheticTransparentCtor { name } => {
                write!(f, "<synthetic-transparent-ctor {name}>")
            }
            CallTarget::Indirect {
                trait_root,
                method_name,
            } => write!(f, "<dyn {trait_root}>::{method_name}"),
            CallTarget::UnsupportedExpr => f.write_str("<unsupported-call-expr>"),
        }
    }
}

/// RPython call ops always carry `op.args[0]` as the funcptr operand.
/// Pyre keeps the same semantic slot but needs two Rust-level shapes:
/// a symbolic direct-call target or a runtime `ValueId` for indirect calls.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CallFuncPtr {
    Target(CallTarget),
    Value(ValueId),
}

/// RPython `flatten.py:53-57`:
///
/// ```python
/// class IndirectCallTargets(object):
///     def __init__(self, lst):
///         self.lst = lst       # list of JitCodes
/// ```
///
/// Sidecar attached to `OpKind::CallResidual` when the residual call is the
/// tail of a regular-indirect lowering (`jtransform.py:547`).  The assembler
/// merges the JitCode handles into `Assembler.indirectcalltargets` so the
/// metainterp can later look up jitcodes by function-pointer address during
/// runtime dispatch (`pyjitpl.py:2325-2343`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndirectCallTargets {
    /// `Arc<JitCode>` shells (identity-keyed via `JitCodeHandle`) for
    /// every candidate impl in the `(trait, method)` family.  Matches
    /// RPython `flatten.py:55` `lst # list of JitCodes` shape.
    pub lst: Vec<crate::jitcode::JitCodeHandle>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FieldDescriptor {
    pub name: String,
    pub owner_root: Option<String>,
}

impl FieldDescriptor {
    pub fn new(name: impl Into<String>, owner_root: Option<String>) -> Self {
        Self {
            name: name.into(),
            owner_root,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpKind {
    Input {
        name: String,
        ty: ValueType,
    },
    ConstInt(i64),
    /// RPython `flowmodel.py:Constant(bool_value)` — a bool constant
    /// whose `concretetype` is `lltype.Bool`. The codewriter folds Bool
    /// into kind `'int'` (`rpython/jit/codewriter/flatten.py:getkind`),
    /// so backend lowering shares the integer materialization path with
    /// `ConstInt`. Distinct from `ConstInt(0/1)` at the annotator /
    /// rtyper layer: lifts to `SomeBool` (`annotator/model.py:227`)
    /// instead of `SomeInteger`, and selects `BoolRepr`
    /// (`rpython/rtyper/rbool.py:10`) instead of `IntegerRepr`.
    ConstBool(bool),
    /// RPython `flowmodel.py:Constant(rfloat)` — a float constant whose
    /// `concretetype` is `lltype.Float`.  Stored as the f64 bit pattern
    /// (`history.py:265 ConstFloat.getfloatstorage`) so PartialEq/Hash
    /// stay derivable.  The assembler materialises this through the
    /// existing `constants_f` pool with a `float_copy` op, mirroring
    /// the `ConstInt` → `int_copy` lowering.
    ConstFloat(u64),
    FieldRead {
        base: ValueId,
        field: FieldDescriptor,
        ty: ValueType,
        /// RPython `jtransform.py:867-903` may rewrite immutable /
        /// quasi-immutable reads to `getfield_*_pure`.  Carries the
        /// chosen opcode flavour through flatten/assembly so the
        /// runtime sees the `_pure` bytecode variant instead of having
        /// to rediscover purity from the descriptor later.
        #[serde(default)]
        pure: bool,
    },
    FieldWrite {
        base: ValueId,
        field: FieldDescriptor,
        value: ValueId,
        ty: ValueType,
    },
    ArrayRead {
        base: ValueId,
        index: ValueId,
        item_ty: ValueType,
        /// RPython: ARRAY identity for `cpu.arraydescrof(ARRAY)`.
        /// Distinguishes arrays with the same item_ty but different
        /// container types (e.g. `Vec<Point>` vs `Vec<Line>`).
        array_type_id: Option<String>,
        /// RPython: `ARRAY_INSIDE._hints.get('nolength', False)`
        /// (descr.py:359). `true` when the operand addresses a raw
        /// items region with no length header; `false` (the common
        /// length-prefixed shape) lays the length word at offset 0
        /// and items past it.
        nolength: bool,
    },
    ArrayWrite {
        base: ValueId,
        index: ValueId,
        value: ValueId,
        item_ty: ValueType,
        /// RPython: ARRAY identity for `cpu.arraydescrof(ARRAY)`.
        array_type_id: Option<String>,
        /// RPython: `ARRAY_INSIDE._hints.get('nolength', False)`
        /// (descr.py:359). See `ArrayRead::nolength`.
        nolength: bool,
    },
    /// RPython: getinteriorfield_gc_i/r/f — read a field of an array-of-structs element.
    /// effectinfo.py:313-325: generates "readinteriorfield" effect.
    /// effectinfo.py:327-340: also implicitly generates "readarray" effect.
    InteriorFieldRead {
        base: ValueId,
        index: ValueId,
        field: FieldDescriptor,
        item_ty: ValueType,
        array_type_id: Option<String>,
    },
    /// RPython: setinteriorfield_gc — write a field of an array-of-structs element.
    /// effectinfo.py:349-350: generates "interiorfield" effect.
    /// effectinfo.py:327-340: also implicitly generates "array" effect.
    InteriorFieldWrite {
        base: ValueId,
        index: ValueId,
        field: FieldDescriptor,
        value: ValueId,
        item_ty: ValueType,
        array_type_id: Option<String>,
    },
    Call {
        target: CallTarget,
        args: Vec<ValueId>,
        result_ty: ValueType,
    },
    GuardTrue {
        cond: ValueId,
    },
    GuardFalse {
        cond: ValueId,
    },

    // ── JIT-specific ops (generated by jtransform pass) ──────────
    /// Guard that a value equals a compile-time constant.
    ///
    /// RPython upstream emits three opnames in the guard_value family
    /// via `jit_codewriter/jtransform.py:608-614 rewrite_op_hint`:
    /// `int_guard_value` / `ref_guard_value` / `float_guard_value`,
    /// each a 1-input/0-output pointer-or-value compare per the arg's
    /// `getkind()`.
    ///
    /// `str_guard_value` (`jit.py:631` for `promote_string`, `:647`
    /// for `promote_unicode`) is NOT modeled here: pyre lacks the
    /// `Ptr(rstr.STR)` / `Ptr(rstr.UNICODE)` GC layouts
    /// (`rpython/rtyper/lltypesystem/rstr.py:1226-1246`) that the upstream
    /// helper (`support.py:526-538 _ll_2_str_eq_nonnull`) indexes via
    /// `s.chars[i]`.  The `PromoteString` / `PromoteUnicode` rewrite
    /// arms panic before reaching emission, so the IR never carries
    /// the 3-input `str_guard_value` shape.
    GuardValue {
        value: ValueId,
        /// `'i'` int, `'r'` ref, `'f'` float — matching the
        /// `<kind>_guard_value` family naming at
        /// `jit_codewriter/jtransform.py:611`.
        kind_char: char,
    },
    /// Project a callee function pointer out of a `dyn Trait` receiver's
    /// vtable for the named method slot.  Result is integer-typed so it
    /// can be fed to `int_guard_value` (RPython `jtransform.py:546`).
    ///
    /// PRE-EXISTING-ADAPTATION of `rclass.py:371-377 getclsfield()`. RPython
    /// emits a `cast_pointer + getfield(vtable_struct, method_name)` chain
    /// because `ClassRepr` models the vtable as an explicit `Struct`. Rust
    /// `dyn Trait` vtable layout is compiler-internal (unstable ABI), so
    /// pyre cannot model the vtable as an IR struct — this single op stands
    /// in for the chain and must be emitted by the rtyper-equivalent layer
    /// (`translator/rtyper/rclass.rs`), never by `jtransform`.
    VtableMethodPtr {
        receiver: ValueId,
        trait_root: String,
        method_name: String,
    },
    /// Indirect call — `op.args[0]` is the funcptr ValueId already produced
    /// by the rtyper layer (e.g. from `VtableMethodPtr` for `dyn Trait`
    /// dispatch). `args` are the full call arguments, including the
    /// receiver. `graphs` mirrors the trailing `c_graphs` constant from
    /// `rpbc.py:216`: `Some(full_family)` when known, `None` otherwise.
    ///
    /// RPython: `rpython/rtyper/rpbc.py:216-217`
    /// ```python
    /// vlist.append(hop.inputconst(Void, row_of_graphs.values()))
    /// v = hop.genop('indirect_call', vlist, resulttype=rresult)
    /// ```
    /// Lowered downstream by `jtransform.py:410-412 rewrite_op_indirect_call`.
    IndirectCall {
        funcptr: ValueId,
        args: Vec<ValueId>,
        graphs: Option<Vec<crate::parse::CallPath>>,
        result_ty: ValueType,
    },
    /// Virtualizable field read → reads from boxes, no heap op.
    /// RPython: `getfield_vable_i/r/f`
    VableFieldRead {
        base: ValueId,
        field_index: usize,
        ty: ValueType,
    },
    /// Virtualizable field write → writes to boxes, no heap op.
    /// RPython: `setfield_vable_i/r/f`
    VableFieldWrite {
        base: ValueId,
        field_index: usize,
        value: ValueId,
        ty: ValueType,
    },
    /// Virtualizable array read → reads from boxes.
    /// RPython: `getarrayitem_vable_i/r/f`
    VableArrayRead {
        base: ValueId,
        array_index: usize,
        elem_index: ValueId,
        item_ty: ValueType,
        /// RPython: arraydescr.itemsize from VirtualizableInfo.array_descrs.
        array_itemsize: usize,
        /// RPython: arraydescr.is_item_signed() from VirtualizableInfo.array_descrs.
        array_is_signed: bool,
    },
    /// Virtualizable array write → writes to boxes.
    /// RPython: `setarrayitem_vable_i/r/f`
    VableArrayWrite {
        base: ValueId,
        array_index: usize,
        elem_index: ValueId,
        value: ValueId,
        item_ty: ValueType,
        /// RPython: arraydescr.itemsize from VirtualizableInfo.array_descrs.
        array_itemsize: usize,
        /// RPython: arraydescr.is_item_signed() from VirtualizableInfo.array_descrs.
        array_is_signed: bool,
    },
    /// Binary arithmetic/comparison operation.
    /// RPython: `int_add`, `int_lt`, etc.
    BinOp {
        op: String,
        lhs: ValueId,
        rhs: ValueId,
        result_ty: ValueType,
    },
    /// Unary operation.
    /// RPython: `int_neg`, `bool_not`, etc.
    UnaryOp {
        op: String,
        operand: ValueId,
        result_ty: ValueType,
    },

    /// Force virtualizable: flush boxes to heap.
    /// RPython: `hint_force_virtualizable(vable)`
    VableForce {
        base: ValueId,
    },

    // ── Call effect classification (generated by jtransform) ────
    //
    // RPython jtransform.py:414-435 `rewrite_call()`: args are split by kind
    // into three ListOfKind lists. The opname encodes the kind signature:
    //   residual_call_ir_i  = int+ref args, int result
    //   call_elidable_r_v   = ref args, void result
    //
    /// Elidable (pure) call — no side effects, result depends only on args.
    /// RPython: `call_elidable_{kinds}_{reskind}(funcptr, calldescr, [i], [r], [f])`
    /// `funcptr` mirrors `op.args[0]` in RPython. Direct calls keep a
    /// symbolic target; indirect calls carry the runtime `ValueId`
    /// produced by rtype.
    CallElidable {
        funcptr: CallFuncPtr,
        descriptor: crate::call::CallDescriptor,
        args_i: Vec<ValueId>,
        args_r: Vec<ValueId>,
        args_f: Vec<ValueId>,
        result_kind: char,
    },
    /// Residual call — has side effects, must be preserved.
    /// RPython: `residual_call_{kinds}_{reskind}(funcptr, calldescr, [i], [r], [f])`.
    /// See `CallElidable` for `funcptr` semantics.
    /// `indirect_targets` mirrors the `IndirectCallTargets(lst)` sidecar
    /// that RPython appends to the extraargs for an indirect_call family
    /// (`jtransform.py:547`).  `None` for direct-call lowering.
    CallResidual {
        funcptr: CallFuncPtr,
        descriptor: crate::call::CallDescriptor,
        args_i: Vec<ValueId>,
        args_r: Vec<ValueId>,
        args_f: Vec<ValueId>,
        result_kind: char,
        indirect_targets: Option<IndirectCallTargets>,
    },
    /// May-force call — can trigger GC or force virtualizables.
    /// RPython: `call_may_force_{kinds}_{reskind}(funcptr, calldescr, [i], [r], [f])`.
    /// See `CallElidable` for `funcptr` semantics.
    CallMayForce {
        funcptr: CallFuncPtr,
        descriptor: crate::call::CallDescriptor,
        args_i: Vec<ValueId>,
        args_r: Vec<ValueId>,
        args_f: Vec<ValueId>,
        result_kind: char,
    },

    // ── Call kind classification (generated by jtransform via CallControl) ──
    //
    // RPython jtransform.py:414-435 `rewrite_call()`: args are split by kind
    // into three lists (int, ref, float). The opname encodes the kind signature:
    //   inline_call_ir_i  = int+ref args, int result
    //   residual_call_r_v = ref args, void result
    //
    // `result_kind`: 'i', 'r', 'f', or 'v' (RPython `getkind(result.concretetype)`)
    /// Inline call — callee is a regular candidate graph.
    /// RPython: `inline_call_{kinds}_{reskind}(jitcode, [i_args], [r_args], [f_args])`
    /// RPython jtransform.py:473-482.
    InlineCall {
        /// RPython: `callcontrol.get_jitcode(targetgraph)` returns the
        /// callee JitCode object itself, not its final `index`.
        /// pyre carries the same identity-bearing handle until the
        /// assembler snapshots the final descriptor table.
        jitcode: crate::jitcode::JitCodeHandle,
        /// Integer arguments (RPython: ListOfKind('int', ...))
        args_i: Vec<ValueId>,
        /// Reference arguments (RPython: ListOfKind('ref', ...))
        args_r: Vec<ValueId>,
        /// Float arguments (RPython: ListOfKind('float', ...))
        args_f: Vec<ValueId>,
        /// Result kind: 'i', 'r', 'f', or 'v'
        result_kind: char,
    },
    /// Recursive call — back to the portal entry point.
    /// RPython: `recursive_call_{reskind}(jd_index, [green_i], [green_r], [green_f], [red_i], [red_r], [red_f])`
    /// RPython jtransform.py:522-534.
    RecursiveCall {
        /// RPython: `jitdriver_sd.index`
        jd_index: usize,
        /// Green args (loop-invariant) split by kind
        greens_i: Vec<ValueId>,
        greens_r: Vec<ValueId>,
        greens_f: Vec<ValueId>,
        /// Red args (loop-variant) split by kind
        reds_i: Vec<ValueId>,
        reds_r: Vec<ValueId>,
        reds_f: Vec<ValueId>,
        /// Result kind
        result_kind: char,
    },

    // ── JIT builtin ops (jtransform.py:1731-1743) ────────────
    //
    // These correspond to RPython's `_handle_jit_call()` in jtransform.py.
    // The codewriter converts calls to `jit.*` oopspec functions into
    // dedicated opcodes instead of residual calls.
    /// jtransform.py:1731 — `jit_debug(string, arg1, arg2, arg3, arg4)`.
    /// Emits debug info into the trace (like debug_merge_point).
    JitDebug {
        args: Vec<ValueId>,
    },
    /// jtransform.py:1733 — `{kind}_assert_green(value)`.
    /// Asserts the value is compile-time constant during tracing.
    AssertGreen {
        value: ValueId,
        kind_char: char,
    },
    /// jtransform.py:1736 — `current_trace_length()`.
    /// Returns the current length of the trace being compiled.
    CurrentTraceLength,
    /// jtransform.py:1738 — `{kind}_isconstant(value)`.
    /// Returns whether the value is currently known to be constant.
    IsConstant {
        value: ValueId,
        kind_char: char,
    },
    /// jtransform.py:1741 — `{kind}_isvirtual(value)`.
    /// Returns whether the value is currently virtualized.
    IsVirtual {
        value: ValueId,
        kind_char: char,
    },

    // ── Conditional call ops (jtransform.py:1665-1688) ──────
    //
    // RPython: `jit_conditional_call` / `jit_conditional_call_value` llops
    // are rewritten to `conditional_call_{kinds}_{reskind}`.
    /// jtransform.py:1685 — `conditional_call_{ir}_{v}`.
    /// If condition is true, call the function. Always produces void.
    /// RPython: `COND_CALL(condition, funcptr, calldescr, args...)`
    ConditionalCall {
        condition: ValueId,
        funcptr: CallTarget,
        descriptor: crate::call::CallDescriptor,
        args_i: Vec<ValueId>,
        args_r: Vec<ValueId>,
        args_f: Vec<ValueId>,
    },
    /// jtransform.py:1687 — `conditional_call_value_{ir}_{reskind}`.
    /// If value is falsy (0/NULL/None), call the function and return its result.
    /// RPython: `COND_CALL_VALUE(value, funcptr, calldescr, args...)`
    ConditionalCallValue {
        value: ValueId,
        funcptr: CallTarget,
        descriptor: crate::call::CallDescriptor,
        args_i: Vec<ValueId>,
        args_r: Vec<ValueId>,
        args_f: Vec<ValueId>,
        result_kind: char,
    },

    /// jtransform.py:292-313 — `record_known_result_{i|r}_ir_v`.
    /// Produced by `rewrite_op_jit_record_known_result`; pairs an elidable call
    /// with its known result for constant folding by OptPure.
    /// RPython layout: `record_known_result_{reskind}(result, funcptr, calldescr, [i], [r])`
    RecordKnownResult {
        /// The known result value (arg 0 of the jit_record_known_result llop).
        result_value: ValueId,
        funcptr: CallTarget,
        descriptor: crate::call::CallDescriptor,
        args_i: Vec<ValueId>,
        args_r: Vec<ValueId>,
        args_f: Vec<ValueId>,
        /// 'i' or 'r' — kind of the known result (no float support).
        result_kind: char,
    },

    /// RPython `record_quasiimmut_field(v_inst, fielddescr, mutatefielddescr)`
    /// — `jtransform.py:901-903`.  Emitted by `rewrite_op_getfield` when the
    /// field is quasi-immutable; paired with a subsequent pure-read.  The
    /// metainterp/blackhole counterpart (`blackhole.py:1537-1539
    /// bhimpl_record_quasiimmut_field`) is a no-op during blackhole execution
    /// but the descriptors drive guard/invalidation accounting in the
    /// optimizer (`quasiimmut.py`).
    ///
    /// PRE-EXISTING-ADAPTATION: RPython derives `mutate_field` via
    /// `quasiimmut.get_mutate_field_name(name)` which expects the lltype
    /// `inst_` prefix (`quasiimmut.py:11-15`).  Rust structs have no such
    /// prefix, so we use the literal `mutate_<fieldname>` convention.
    RecordQuasiImmutField {
        base: ValueId,
        field: FieldDescriptor,
        mutate_field: FieldDescriptor,
    },

    /// Liveness marker — RPython `-live-` operation.
    /// Inserted by jtransform after calls that may need guard resumption.
    /// Expanded by compute_liveness() to include all values alive at
    /// this point. RPython: jtransform.py:469,481,533.
    Live,

    /// JitDriver merge-point marker — RPython `jit_merge_point` opname.
    /// Emitted by `handle_jit_marker__jit_merge_point` (jtransform.py:1690-1712)
    /// with the portal jitdriver's index and green/red arguments split by
    /// kind (`make_three_lists`). In upstream the args are a flat
    /// `SpaceOperation.args` vec `[index_const, greens_i, greens_r,
    /// greens_f, reds_i, reds_r, reds_f]`; pyre stores them as structured
    /// fields to avoid re-splitting on every consumer.
    JitMergePoint {
        jitdriver_index: usize,
        greens_i: Vec<ValueId>,
        greens_r: Vec<ValueId>,
        greens_f: Vec<ValueId>,
        reds_i: Vec<ValueId>,
        reds_r: Vec<ValueId>,
        reds_f: Vec<ValueId>,
    },

    /// JitDriver loop-header marker — RPython `loop_header` opname.
    /// Emitted by `handle_jit_marker__loop_header` (jtransform.py:1714-1718)
    /// with the jitdriver's index as its single Constant arg.
    /// `can_enter_jit` markers alias to this (jtransform.py:1723).
    LoopHeader {
        jitdriver_index: usize,
    },

    /// pyre-only marker emitted by the front-end (`front/ast.rs`
    /// `continue_with_unknown*` / `stop_unsupported`) when a syntactic
    /// form cannot be lowered to a canonical opname.  Reaching the op at
    /// runtime means tracing or blackhole resume crossed an
    /// untranslatable graph slice; downstream handlers advance past it
    /// (see `blackhole.rs::handler_abort_marker_pyre`).  Distinct from
    /// RPython's `SwitchToBlackhole` exception path — RPython aborts
    /// before lowering so no equivalent opname exists.  Kept under
    /// `kind: UnknownKind` because the same diagnostic enum is reused
    /// by `FlowingError::Unsupported` (`front/ast.rs:53`).
    Abort {
        kind: UnknownKind,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpaceOperation {
    pub result: Option<ValueId>,
    pub kind: OpKind,
}

/// RPython `Block.exitswitch`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExitSwitch {
    Value(ValueId),
    LastException,
}

/// RPython `Link.exitcase`.
///
/// Upstream stores the concrete switch value itself here: `False` /
/// `True` for boolean branches, the Python `Exception` class object for
/// catch-all exception links, or a specific exception class object for
/// typed handlers (`flowspace/model.py:114-120`,
/// `flowspace/flowcontext.py:127-143`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExitCase {
    Bool(bool),
    Const(ConstValue),
}

/// RPython `flowspace/model.py:109-168` `Link`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Link {
    pub args: Vec<LinkArg>,
    pub target: BlockId,
    pub exitcase: Option<ExitCase>,
    /// RPython `Link.prevblock` — the block this Link exits from.
    pub prevblock: Option<BlockId>,
    /// RPython `Link.llexitcase` — the low-level value matched by
    /// `goto_if_exception_mismatch` (`flatten.py:228-231`).  For
    /// typed exception links this is the rtyper-produced class
    /// identity constant; pyre carries it as a full `ConstValue` so
    /// non-Int llexitcase shapes (`lltype.Ptr`, host class objects)
    /// round-trip to the backend intact.
    pub llexitcase: Option<ConstValue>,
    pub last_exception: Option<LinkArg>,
    pub last_exc_value: Option<LinkArg>,
}

impl Link {
    pub fn new(args: Vec<ValueId>, target: BlockId, exitcase: Option<ExitCase>) -> Self {
        Self::new_mixed(
            args.into_iter().map(LinkArg::from).collect(),
            target,
            exitcase,
        )
    }

    pub fn new_mixed(args: Vec<LinkArg>, target: BlockId, exitcase: Option<ExitCase>) -> Self {
        Self {
            args,
            target,
            exitcase,
            prevblock: None,
            llexitcase: None,
            last_exception: None,
            last_exc_value: None,
        }
    }

    pub fn with_prevblock(mut self, prevblock: BlockId) -> Self {
        self.prevblock = Some(prevblock);
        self
    }

    pub fn with_llexitcase(mut self, llexitcase: ConstValue) -> Self {
        self.llexitcase = Some(llexitcase);
        self
    }

    /// Structural counterpart of RPython rtyper's
    /// `convert_link()` (`rtyper.py:1338`): copy the flow-level
    /// `exitcase` into the low-level `llexitcase` slot for primitive
    /// branch values.  The `"default"` switch sentinel is not a
    /// low-level case and stays `None`.
    pub fn with_llexitcase_from_exitcase(mut self) -> Self {
        self.llexitcase = match &self.exitcase {
            Some(ExitCase::Bool(value)) => Some(ConstValue::Bool(*value)),
            Some(ExitCase::Const(value)) if value.string_eq("default") => None,
            Some(ExitCase::Const(value)) => Some(value.clone()),
            None => None,
        };
        self
    }

    pub fn extravars(
        mut self,
        last_exception: Option<LinkArg>,
        last_exc_value: Option<LinkArg>,
    ) -> Self {
        self.last_exception = last_exception;
        self.last_exc_value = last_exc_value;
        self
    }

    /// RPython `flatten.py:224` `if link.exitcase is Exception`.
    pub fn catches_all_exceptions(&self) -> bool {
        self.exitcase == Some(exception_exitcase())
    }
}

pub fn exception_exitcase() -> ExitCase {
    ExitCase::Const(ConstValue::builtin("Exception"))
}

/// RPython `Link.args` items are Variables or Constants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkArg {
    Value(ValueId),
    Const(ConstValue),
}

impl LinkArg {
    pub fn as_value(&self) -> Option<ValueId> {
        match self {
            Self::Value(value) => Some(*value),
            Self::Const(_) => None,
        }
    }
}

impl From<ValueId> for LinkArg {
    fn from(value: ValueId) -> Self {
        Self::Value(value)
    }
}

impl From<ConstValue> for LinkArg {
    fn from(value: ConstValue) -> Self {
        Self::Const(value)
    }
}

/// A basic block in the control flow graph.
///
/// RPython equivalent: `flowspace/model.py:171-180 Block` — slots
/// `inputargs operations exitswitch exits`.  Upstream has no separate
/// terminator surface: fall-through is `exitswitch=None` with a single
/// `Link`, bool branches are `exitswitch=Variable` with two Links
/// carrying `Bool(false)`/`Bool(true)` exitcases, can-raise is
/// `exitswitch=c_last_exception`, and final blocks have `exits=()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub id: BlockId,
    /// Phi-node inputs: values provided by incoming Links.
    /// RPython: `Block.inputargs` — each predecessor Link carries
    /// values that map 1:1 to these inputargs.
    pub inputargs: Vec<ValueId>,
    pub operations: Vec<SpaceOperation>,
    /// RPython `Block.exitswitch`.
    pub exitswitch: Option<ExitSwitch>,
    /// RPython `Block.exits`.
    pub exits: Vec<Link>,
    /// RPython `flowspace/flowcontext.py:455 mergeblock`: when a
    /// `SpamBlock` is generalised by a later `union`, the prior
    /// candidate is marked `block.dead = True` and `block.operations
    /// = ()` and its exits are recloseblock'd to forward to the new
    /// SpamBlock.  `simplify.eliminate_empty_blocks` then collapses
    /// the dead-block forwarding chain into a single multi-incoming
    /// merge block.  Pyre's tree-recursive `Expr::Match` /
    /// `Expr::If` lowering produces the post-collapse CFG directly
    /// today (no chain to collapse), so the field is always `false`
    /// in the AST graph; it becomes load-bearing once the Z4
    /// flowcontext-walker rewrite materialises intermediate
    /// SpamBlocks per fold step.
    pub dead: bool,
    /// Framestate snapshot captured at the moment this block was closed
    /// (its `set_goto` / `set_branch` / `set_return`).  Read by the
    /// frontend's lazy cross-block local installer to recover, after the
    /// fact, what `(name → ValueId)` mapping was visible to a
    /// predecessor when its outgoing link fired.  Build-time only —
    /// downstream rtyper/jtransform passes ignore this field; it is
    /// neither serialised nor required to remain populated past the
    /// front end.  RPython parity: `flowspace/flowcontext.py:38
    /// SpamBlock.__init__ self.framestate = framestate` attaches a
    /// framestate to the block representing the locals state visible at
    /// that block boundary; pyre stores the analogous "exit-time"
    /// snapshot here so cross-block reads in successors can derive
    /// `Link.args` retroactively.
    pub framestate: Option<FrameState>,
}

/// Locals snapshot captured at a block boundary — see `Block.framestate`
/// for usage semantics.  Entries are stored **densely** at graph-wide
/// first-bind slot positions: slot `i` corresponds to the name at
/// `GraphBuildContext::local_first_bind_order[i]`, and `None` indicates
/// the slot is unbound at this snapshot point (RPython's "undefined
/// local" sentinel).  Names are appended to the graph-wide order on
/// first bind anywhere in the function and never moved or removed —
/// `LocalBindingSnapshot::restore` does NOT roll the order back, so the
/// slot index of a name is truly invariant across the entire lowering.
/// This mirrors RPython `co_varnames` slot order — every predecessor's
/// exit snapshot walks the same slot positions, so two predecessors of
/// the same merge point line up positionally even when one of them
/// rolled back its bindings via `LocalBindingSnapshot::restore`.
///
/// RPython parity: `flowspace/framestate.py:18 FrameState` — a tuple of
/// `(locals_w, stack, last_exception, blocklist, next_offset)`.  Pyre's
/// AST frontend currently runs over Rust source rather than Python
/// bytecode, so the `stack` / `last_exception` / `blocklist` /
/// `next_offset` projections are vestigially empty until Path-Z Slice
/// 4+ rewrites `front::ast` as a flowcontext-style walker.
///
/// All five fields are present in the struct now (Path-Z Slice 1) so
/// shape parity is locked in at the model layer; downstream merging
/// passes already thread them via `union`.  The flowcontext walker
/// (`flowspace::flowcontext::FlowContext`) and `flowspace::framestate::
/// FrameState` are the upstream-orthodox surfaces this struct is
/// converging toward — eventually `front::ast` produces flowspace
/// graphs directly and this AST-shaped `FrameState` retires.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FrameState {
    /// Slot `i` ↔ graph-wide first-bind name at index `i`; `None` if
    /// unbound at this snapshot point.  RPython parity:
    /// `framestate.py:19 self.locals_w` — list of `Variable | Constant
    /// | None` indexed by `co_varnames` slot.  Pyre's AST-frontend
    /// graph keys variable identity through `ValueId`; Path-Z Slice 2.5
    /// replaces this with `Vec<Option<Hlvalue>>` once the front::ast
    /// cutover lands, dropping the AST-side identity in favour of
    /// upstream `flowspace::Variable`.
    pub entries: Vec<Option<ValueId>>,
    /// `framestate.py:21 self.stack` — value-stack content at the
    /// snapshot point.  Empty for AST-frontend snapshots until Path-Z
    /// Slice 4+ introduces flowcontext-style stack push/pop on Expr
    /// nodes; the `union` invariant requires both predecessors agree
    /// on stack content (upstream `framestate.py:79 _union(self.stack,
    /// other.stack)` zips positionally — equal-length always for any
    /// program reachable to a join).
    pub stack: Vec<crate::flowspace::framestate::StackElem>,
    /// `framestate.py:22 self.last_exception` — pending FSException at
    /// the snapshot point.  None for AST-frontend snapshots until
    /// Path-Z Slice 4+ introduces flowcontext-style exception handling.
    pub last_exception: Option<crate::flowspace::model::FSException>,
    /// `framestate.py:23 self.blocklist` — block-stack snapshot
    /// (`SETUP_*` / `POP_BLOCK` depth at the snapshot point).  Empty
    /// for AST-frontend snapshots until Path-Z Slice 4+ introduces
    /// frame-block management.  `framestate.py:58 matches` asserts
    /// blocklist equality across merge candidates as a precondition.
    pub blocklist: Vec<crate::flowspace::flowcontext::FrameBlock>,
    /// `framestate.py:24 self.next_offset` — bytecode offset resumed
    /// at after the snapshot.  `0` for AST-frontend snapshots until
    /// Path-Z Slice 4+ uses an AST-node index (the equivalent of a
    /// virtual-bytecode tape position).  `framestate.py:59 matches`
    /// asserts next_offset equality across merge candidates as a
    /// precondition.
    pub next_offset: i64,
}

impl FrameState {
    /// Iterate `(slot_idx, ValueId)` over **bound** slots in storage
    /// order.  Unbound (None-killed) slots are skipped.  Callers
    /// translate `slot_idx` → name via
    /// `GraphBuildContext::local_first_bind_order[slot_idx]` and
    /// query the type via `graph_value_type(graph, value_id)` —
    /// upstream `framestate.py:locals_w` slot-index convention with
    /// type carried on the `Variable.concretetype` slot.
    pub fn iter(&self) -> impl Iterator<Item = (usize, ValueId)> + '_ {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| e.map(|vid| (i, vid)))
    }

    /// `Link.args` payload for this snapshot — `ValueId`s of bound
    /// slots in storage order.  Unbound (None) slots are skipped.
    #[allow(dead_code)]
    pub fn link_args(&self) -> Vec<ValueId> {
        self.entries.iter().filter_map(|e| *e).collect()
    }

    /// Compute a state at least as general as both `self` and `other`
    /// via positional zip over slots — direct port of RPython
    /// `framestate.py:14 _union` (`return [union(v1, v2) for v1, v2
    /// in zip(seq1, seq2)]`) and `framestate.py:73-90 FrameState.union`.
    /// When one side has fewer slots than the other (a name appended
    /// after the shorter side's snapshot was taken), the missing
    /// slots are treated as `None` so the positional zip extends to
    /// `max(len(self), len(other))`.
    ///
    /// Per-slot semantics (RPython `framestate.py:105-128 union`):
    ///   - `(None, _) | (_, None)` → None-kill (`framestate.py:
    ///     110-111`); merged slot is `None`.
    ///   - `(Some(s), Some(o))` with `s.value_id == o.value_id` →
    ///     carry through that vid (`framestate.py:108 if w1 == w2:
    ///     return w1`).
    ///   - `(Some(s), Some(o))` with disagreeing vids →
    ///     `graph.alloc_value()` for a fresh `ValueId` at the slot
    ///     (`framestate.py:113-114 return Variable()` analogue).
    ///
    /// Type unification is NOT performed here — upstream's per-slot
    /// `union(w1, w2)` (`framestate.py:105-128`) compares Hlvalue
    /// identity only, with type-side reconciliation deferred to the
    /// annotator (`annrpython.py`).  Pyre follows the same convention:
    /// callers query types via `graph_value_type(graph, value_id)` at
    /// the point of use; the prior per-slot `value_type` field on
    /// `FrameStateEntry` was a NEW-DEVIATION that has been retired in
    /// Path-Z Slice 2.3.
    ///
    /// Returns `Some(merged_state)` when the union succeeds; `None`
    /// when any per-projection union raises `UnionError`
    /// (`framestate.py:78 try: ... except UnionError: return None`).
    /// The `None` return propagates upstream's "merge candidates
    /// disagree, fall back to a fresh SpamBlock" path at
    /// `flowcontext.py:431-436 mergeblock`.  Agreement slots in the
    /// locals projection carry the predecessor's vid, disagreement
    /// slots get a freshly-allocated vid (the upstream `Variable()`
    /// analogue).  Callers detect "this slot is a fresh phi" by
    /// comparing the merged `value_id` against the predecessor's vid
    /// for the same slot.  The install is then driven via
    /// `lazy_install_local_at_current_block(.., Some(merged_vid))` so
    /// the Input op carries the same vid the merged state already
    /// refers to.
    pub fn union(&self, other: &FrameState, graph: &mut FunctionGraph) -> Option<FrameState> {
        // Line-by-line port of `framestate.py:73-90 FrameState.union`:
        //
        //     def union(self, other):
        //         try:
        //             locals = _union(self.locals_w, other.locals_w)
        //             stack = _union(self.stack, other.stack)
        //             if self.last_exception is None and other.last_exception is None:
        //                 exc = None
        //             else:
        //                 args1 = self._exc_args()
        //                 args2 = other._exc_args()
        //                 exc = FSException(union(args1[0], args2[0]),
        //                         union(args1[1], args2[1]))
        //         except UnionError:
        //             return None
        //         return FrameState(locals, stack, exc, self.blocklist, self.next_offset)
        //
        // Upstream `union` does NOT compare `self.blocklist` /
        // `self.next_offset` against `other.*` — those equality
        // assertions live in `framestate.py:53-59 matches`, which is a
        // separate predicate the caller invokes AFTER union.  Pyre's
        // AST frontend keeps both projections at trivial defaults
        // today (empty Vec, next_offset 0); even when they diverge,
        // upstream still produces a merged FrameState that just
        // carries `self.{blocklist,next_offset}` and lets `matches`
        // catch the mismatch downstream.
        //
        // Body order (`framestate.py:79-87`): locals → stack →
        // exception, all inside the try/except envelope.  Pyre
        // **reorders** to (stack → exception → locals): the locals
        // fold is total (ValueId domain has no UnionError analogue)
        // but invokes `graph.alloc_value()` which advances a global
        // allocator counter.  Stack and exception unions CAN return
        // `None`/`Err` once the Z4 walker activates real cells; doing
        // them first keeps `union` atomic — no `alloc_value` writes
        // unless the whole merge succeeds.  Upstream's `Variable()`
        // construction is side-effect-free (per-instance object), so
        // the reorder is a Rust-side atomic-safety adaptation with
        // no observable upstream-parity divergence.

        // `framestate.py:80 stack = _union(self.stack, other.stack)`.
        // `flowspace::framestate::union_stack` returns `Err(UnionError)`
        // on stack length disagreement, SpecTag mismatch, or
        // FlowSignal-type mismatch — propagate as `None` per upstream's
        // try/except envelope (`framestate.py:78,88-89`).
        let stack = crate::flowspace::framestate::union_stack(&self.stack, &other.stack).ok()?;
        // `framestate.py:81-87`: both `last_exception` None → None;
        // otherwise FSException carries `union(args1[i], args2[i])` for
        // `(w_type, w_value)`.  Per-cell `union` returns `Err(UnionError)`
        // on SpecTag mismatch — propagate as `None`.  When `union()`
        // returns `Ok(None)` for an exception slot (one side undefined-
        // local), we propagate as `None` too — exception slot `None`
        // is not a valid merge result because both sides carry
        // `Constant(None)` sentinels through `_exc_args`, so an
        // undefined-local result here would only arise from a sentinel-
        // vs-non-sentinel mix, which signals the same kind of merge
        // refusal upstream returns None for.
        let last_exception = if self.last_exception.is_none() && other.last_exception.is_none() {
            None
        } else {
            let a = exc_args(&self.last_exception);
            let b = exc_args(&other.last_exception);
            let w_type = crate::flowspace::framestate::union(Some(&a[0]), Some(&b[0])).ok()??;
            let w_value = crate::flowspace::framestate::union(Some(&a[1]), Some(&b[1])).ok()??;
            Some(crate::flowspace::model::FSException::new(w_type, w_value))
        };
        // `framestate.py:79 locals = _union(self.locals_w, other.locals_w)`.
        // Run LAST so `graph.alloc_value()` only fires when the
        // failure-prone projections above have succeeded.
        //
        // Pyre's `entries` carry `Option<ValueId>` (graph-local Hlvalue
        // identity surrogate) instead of upstream `Option<Hlvalue>`;
        // until Z2.5 (Hlvalue migration, subsumed into Z4.last) lands
        // the per-slot match below stands in for upstream's polymorphic
        // `union(w1, w2)` (`framestate.py:105-128`), specialised to the
        // ValueId-identity domain — the ValueId-equal carry-through
        // mirrors `if w1 == w2: return w1`, the None-kill mirrors
        // `if w1 is None or w2 is None: return None`, and the fresh
        // alloc mirrors `return Variable()` for unequal-but-defined
        // cells.  ValueId-vs-ValueId never raises UnionError (no
        // SpecTag analogue in pyre's locals projection), so the
        // locals fold is total.
        let len = std::cmp::max(self.entries.len(), other.entries.len());
        let merged: Vec<Option<ValueId>> = (0..len)
            .map(|i| {
                let s = self.entries.get(i).copied().flatten();
                let o = other.entries.get(i).copied().flatten();
                match (s, o) {
                    (None, _) | (_, None) => None,
                    (Some(s_vid), Some(o_vid)) if s_vid == o_vid => Some(s_vid),
                    (Some(_), Some(_)) => Some(graph.alloc_value()),
                }
            })
            .collect();
        Some(FrameState {
            entries: merged,
            stack,
            last_exception,
            blocklist: self.blocklist.clone(),
            next_offset: self.next_offset,
        })
    }

    /// Output arguments to thread `self` (a predecessor's exit state)
    /// into the merge block whose entry framestate is `target`.  RPython
    /// parity: `framestate.py:92-99 FrameState.getoutputargs` walks
    /// `targetstate.mergeable` positionally, picking `self.mergeable[i]`
    /// at every Variable position.  `target.entries` is dense at the
    /// graph-wide first-bind slot positions, with `None` at slots that
    /// were None-killed by the union; surviving (`Some`) slot `i` lines
    /// up with `self.entries[i]` because both predecessors share the
    /// same graph-wide first-bind order.
    ///
    /// Panics if `target` contains a `Some` slot at an index where
    /// `self.entries[i]` is `None` or out of range — that would mean
    /// `target` was not produced by `self.union(_)`, which violates the
    /// merge invariant.
    pub fn getoutputargs(&self, target: &FrameState) -> Vec<ValueId> {
        // Line-by-line port of `framestate.py:92-99 getoutputargs`:
        //
        //     def getoutputargs(self, targetstate):
        //         result = []
        //         mergeable = self.mergeable
        //         for i, w_target in enumerate(targetstate.mergeable):
        //             if isinstance(w_target, Variable):
        //                 result.append(mergeable[i])
        //         return result
        //
        // Upstream's `mergeable` is `locals_w + recursively_flatten(
        // stack) + [exc_type, exc_value]`.  Pyre walks the same three
        // projections in the same order so the positional mapping
        // between target and self is preserved across the locals→
        // stack→exception boundary.
        //
        // Pyre's `entries` (locals) carry `Option<ValueId>` instead of
        // upstream `Option<Hlvalue>`; the Variable predicate becomes
        // `slot.is_some()` because every defined ValueId stands in
        // for upstream `Variable`.
        //
        // Pyre's `stack` and exception args carry `StackElem` /
        // `Hlvalue` cells (Hlvalue identity, NOT ValueId-identity).
        // The AST frontend keeps both empty / sentinel-None today,
        // so the per-cell walks below contribute zero entries.
        //
        // PRE-EXISTING-ADAPTATION (`framestate.py:92-99 getoutputargs`
        // pushes `self.mergeable[i]` — an `Hlvalue` cell — at every
        // `Variable` position): pyre's result type is `Vec<ValueId>`
        // and the Hlvalue→ValueId bridge is being built out by the
        // Z2.5 multi-session epic:
        //   - Slice Z2.5.A (DONE) — `FunctionGraph.variable_to_vid` +
        //     `bridge_variable()` scaffolding, no production wiring.
        //   - Slice Z2.5.B — getoutputargs stack walk consumes the
        //     bridge for `Hlvalue::Variable` cells.  Signature gains
        //     `&mut FunctionGraph` for `bridge_variable` access.
        //   - Slice Z2.5.C — Constant-in-stack/exc → synthetic
        //     `Constant` op + ValueId via `bridge_constant`.
        //   - Slice Z2.5.D — exception projection consumes the bridge
        //     and this marker block is retired.
        // Today's AST frontend keeps both stack and exception cells
        // empty / sentinel-None, so the per-cell walks below
        // contribute zero entries at runtime regardless of the
        // bridge state; the explicit documented skip + Z4 audit point
        // is what gates honest activation of the Z4.B+ walker.
        let mut result: Vec<ValueId> = Vec::new();
        // (1) locals projection — `framestate.mergeable` head.
        for (i, slot) in target.entries.iter().enumerate() {
            if slot.is_some() {
                result.push(
                    self.entries
                        .get(i)
                        .copied()
                        .flatten()
                        .expect("target slot must be bound in self — union invariant"),
                );
            }
        }
        // (2) stack projection — `recursively_flatten(stack)` middle
        // segment.  Walk the flattened stack in step with `self`'s
        // flattened stack so position-`i` lines up between the two
        // mergeable views.  AST-frontend snapshots keep this segment
        // empty; the Z4 walker activates real cells.
        let target_flat_stack = crate::flowspace::framestate::recursively_flatten(&target.stack);
        let self_flat_stack = crate::flowspace::framestate::recursively_flatten(&self.stack);
        for (i, w_target) in target_flat_stack.iter().enumerate() {
            if matches!(w_target, crate::flowspace::model::Hlvalue::Variable(_)) {
                // PRE-EXISTING-ADAPTATION: see block above.  Touching
                // `self_flat_stack[i]` keeps the positional invariant
                // honest; pushing is blocked on the Hlvalue→ValueId
                // bridge (Z2.5 absorption at Z4.last).
                let _w_self = self_flat_stack
                    .get(i)
                    .expect("target stack length must match self stack length — union invariant");
            }
        }
        // (3) exception args projection — `[exc_type, exc_value]`
        // tail.  `exc_args` substitutes `Constant(None)` sentinels
        // when no exception is pending; these are never `Variable`,
        // hence contribute nothing.  The walk is structurally still
        // present (commented) so the positional invariant maps to
        // upstream `framestate.py:34-39 mergeable`'s `[w_type,
        // w_value]` tail when Z4 walker activates real cells with
        // the same Z2.5 bridge as the stack walk above
        // (PRE-EXISTING-ADAPTATION).
        let _target_exc = exc_args(&target.last_exception);
        let _self_exc = exc_args(&self.last_exception);
        result
    }
}

/// RPython `framestate.py:66-71 _exc_args` — return `[w_type, w_value]`,
/// substituting `Constant(None)` sentinels when no exception is
/// pending.
fn exc_args(
    last_exception: &Option<crate::flowspace::model::FSException>,
) -> [crate::flowspace::model::Hlvalue; 2] {
    use crate::flowspace::model::{Constant, Hlvalue};
    match last_exception {
        None => [
            Hlvalue::Constant(Constant::new(ConstValue::None)),
            Hlvalue::Constant(Constant::new(ConstValue::None)),
        ],
        Some(exc) => [exc.w_type.clone(), exc.w_value.clone()],
    }
}

/// RPython `eliminate_empty_blocks(graph)`
/// (`rpython/translator/simplify.py:52-69`).
///
/// ```python
/// def eliminate_empty_blocks(graph):
///     for link in list(graph.iterlinks()):
///         while not link.target.operations:
///             block1 = link.target
///             if block1.exitswitch is not None:
///                 break
///             if not block1.exits:
///                 break
///             exit = block1.exits[0]
///             assert block1 is not exit.target, \
///                 "the graph contains an empty infinite loop"
///             subst = dict(zip(block1.inputargs, link.args))
///             link.args = [v.replace(subst) for v in exit.args]
///             link.target = exit.target
/// ```
///
/// Pyre's tree-recursive `Expr::Match` / `Expr::If` lowering produces
/// the post-collapse CFG directly (no chain to collapse), so this
/// pass is a no-op on production graphs today.  It becomes
/// load-bearing once Z4's flowcontext-walker rewrite materialises
/// intermediate `SpamBlock`s per fold step (the upstream
/// `flowcontext.py:443-463 mergeblock` chain pattern).
pub fn eliminate_empty_blocks(graph: &mut FunctionGraph) {
    use std::collections::HashMap;
    // upstream: `for link in list(graph.iterlinks()):` — walk every
    // (block_id, exit_idx) pair.
    let block_count = graph.blocks.len();
    for block_id_idx in 0..block_count {
        let block_id = BlockId(block_id_idx);
        let exit_count = graph.block(block_id).exits.len();
        for exit_idx in 0..exit_count {
            // upstream: `while not link.target.operations:` — walk
            // through the dead-block chain.
            loop {
                let target = graph.block(block_id).exits[exit_idx].target;
                let target_block = graph.block(target);
                if !target_block.operations.is_empty() {
                    break;
                }
                if target_block.exitswitch.is_some() {
                    break;
                }
                if target_block.exits.is_empty() {
                    break;
                }
                // upstream: `exit = block1.exits[0]`.
                let target_inputargs = target_block.inputargs.clone();
                let target_exit = target_block.exits[0].clone();
                // upstream: `assert block1 is not exit.target, "the
                // graph contains an empty infinite loop"` (`simplify.py:64`).
                assert!(
                    target_exit.target != target,
                    "the graph contains an empty infinite loop: {:?}",
                    target
                );
                // upstream: `subst = dict(zip(block1.inputargs,
                // link.args))`.
                let link_args = graph.block(block_id).exits[exit_idx].args.clone();
                let subst: HashMap<ValueId, LinkArg> = target_inputargs
                    .into_iter()
                    .zip(link_args.into_iter())
                    .collect();
                // upstream: `link.args = [v.replace(subst) for v in
                // exit.args]`.
                let new_args: Vec<LinkArg> = target_exit
                    .args
                    .iter()
                    .map(|arg| match arg {
                        LinkArg::Value(v) => subst.get(v).cloned().unwrap_or_else(|| arg.clone()),
                        LinkArg::Const(_) => arg.clone(),
                    })
                    .collect();
                // upstream: `link.target = exit.target`.
                let link = &mut graph.block_mut(block_id).exits[exit_idx];
                link.args = new_args;
                link.target = target_exit.target;
            }
        }
    }
}

/// Remove dead operations and dead inputargs from `graph` per
/// backward dataflow over operation operands + exitswitches +
/// `Link.args`-as-dependencies.  Line-by-line port of
/// `simplify.transform_dead_op_vars_in_blocks(blocks, graphs,
/// translator=None)` (`rpython/translator/simplify.py:422-524`).
///
/// `blocks` is the BFS-reachable closure of every entry block (mirrors
/// `flowspace/model.py:66 iterblocks()`).
///
/// PRE-EXISTING-ADAPTATION: `start_blocks` is `{graph.startblock} ∪
/// {blocks with no incoming link}` rather than the strict single-graph
/// `{graphs[0].startblock}` of `simplify.py:428`.  The `generated::*`
/// pipeline emits closures / specialisations as secondary entries
/// (function parameters at a non-zero block id with no predecessors)
/// that are calling-convention contracts but are not registered with
/// the simplification pass as separate graphs.  Pinning them as
/// additional starts matches the *intent* of `simplify.py:431-433`'s
/// multi-graph branch (which pins one start per graph in the input
/// set) without requiring a `translator.annotator` to enumerate the
/// secondary entries.  Convergence: surface the secondary entries via
/// an explicit `start_blocks: HashSet<BlockId>` parameter or migrate
/// `generated::*` to register secondary entries with the simplifier
/// directly.
///
///   1. Walk every reachable block.  For each operation, evaluate
///      `canremove(op, block)` per `simplify.py:435-436`:
///      `op.opname in CanRemove and op is not block.raising_op`,
///      where `block.raising_op` is `block.operations[-1]` whenever
///      `block.canraise` (`flowspace/model.py:218-221`).  Pyre's
///      `Block::canraise()` mirrors the upstream
///      `block.exitswitch is c_last_exception` check; the raising
///      op is the last entry in `block.operations`.
///        - If `is_pure_op(kind)` AND the op is not the raising_op
///          AND `op.result.is_some()`: route operands via
///          `dependencies[op.result] += operands`
///          (`simplify.py:444-445 dependencies[op.result].
///          update(op.args)` for `canremove`-classified ops).
///          Operands become live only if `op.result` becomes live.
///        - Otherwise: operands join `read_vars` immediately
///          (`simplify.py:442-443 read_vars.update(op.args)`
///          for non-`canremove` ops).  Raising-op operands therefore
///          stay live unconditionally — the raise side-effect is
///          observable.
///      The `exitswitch` (if a Variable) joins `read_vars`.  For
///      each block with no exits (`returnblock` / `exceptblock` /
///      otherwise terminal), the inputargs are also added — return
///      and except blocks implicitly use their inputs
///      (`simplify.py:459-462`).
///   2. For every Link, record `dependencies[targetarg].add(linkarg)`
///      mapping (`simplify.py:457-458`).  This is the key
///      difference from a naïve forward pass — link args are NOT
///      direct reads, they are dependencies whose liveness flows
///      backward from a live target inputarg.  A self-referential
///      phi (back-edge `Link.args[i] == phi_i`) is therefore kept
///      alive only if `phi_i` itself is reached by some external
///      reader; without one, the dependency cycle dies.
///   3. The `startblock`'s inputargs are pinned to `read_vars` as a
///      calling-convention contract (`simplify.py:463-467`,
///      single-graph case `simplify.py:428-429 start_blocks =
///      {graphs[0].startblock}`).  `returnblock` / `exceptblock`
///      contracts are already covered by the empty-exits rule
///      above.
///   4. Backward-flow `read_vars` over `dependencies` to a fixpoint
///      (`simplify.py:471-479 flow_read_var_backward`).
///   5. Drop every pure op whose result is not in `read_vars`
///      (`simplify.py:484-488` removable-op drop), gated by the
///      same `canremove(op, block)` predicate from Step 1 — the
///      raising op of a `canraise` block is preserved here even if
///      its result is dead.  Inputarg-shaped `OpKind::Input` ops
///      are protected by Step 1+3+dependency-routing pinning their
///      result vid in `read_vars`; the dead ones are swept here
///      alongside Step 7's inputarg trim (the matching
///      `block.inputargs[i]` removal becomes a no-op when the
///      Input op is already gone).  Naked `OpKind::Input` ops
///      (legacy frontend fallback for cross-block reads not yet
///      routed through the lazy installer) are likewise removed
///      when their result vid is dead — `simplify.py` itself has
///      no standalone Input shape, but the line-by-line port
///      sweeps them under the same canremove rule.
///
///      PRE-EXISTING-ADAPTATION (translator-gated arms not ported).
///      Upstream's Step 5 has two further `elif` arms after the
///      canremove drop:
///        - `simplify.py:489-499 elif op.opname == 'simple_call'`
///          removes `simple_call(builtin, ...)` whose first arg is
///          a `Constant` whose value is in `CanRemoveBuiltins`
///          (`simplify.py:418-420 = {hasattr: True}`).  The
///          `FunctionGraph` IR pyre's prune_dead_phis operates on
///          carries call shapes as `OpKind::Call` /
///          `OpKind::IndirectCall` etc., not as `simple_call`-named
///          ops, and pyre's frontend never emits a Call to `hasattr`
///          — the arm is structurally inapplicable.
///        - `simplify.py:500-506 elif op.opname == 'direct_call'`
///          removes the call when `has_no_side_effects(translator,
///          graph) and op is not block.raising_op`.  Gated on
///          `translator is not None`; pyre's caller passes
///          `translator=None`, so the arm is unreachable.  The
///          `op is not block.raising_op` clause is moot for the
///          same reason.
///      Convergence path: surface a translator-equivalent (callee
///      effect classifier) and route `OpKind::Call`-with-elidable-EI
///      through this Step's drop path.
///   6. For each reachable block, walk its exits and drop
///      `Link.args[i]` for every `i` where `target.inputargs[i]`
///      is not in `read_vars` — inputarg-side trim happens in a
///      second pass to preserve the `len(link.args) ==
///      len(link.target.inputargs)` invariant (`simplify.py:
///      512-516`, the explicit assertion at `:513`).
///   7. Walk each reachable non-terminal block once more and drop
///      `block.inputargs[i]` (and its matching `OpKind::Input` op
///      if still present in `block.operations`) for every dead
///      vid (`simplify.py:520-524`).
pub fn prune_dead_phis(graph: &mut FunctionGraph) {
    use crate::inline::{is_pure_op, op_value_refs};
    use std::collections::HashMap;
    let start = graph.startblock;
    let return_block = graph.returnblock;
    let except_block = graph.exceptblock;
    let block_index: HashMap<BlockId, usize> = graph
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect();

    // `start_blocks` mirrors `simplify.py:431-433`'s multi-graph branch:
    // `graph.startblock` plus every block with no incoming link
    // (calling-convention entries the `generated::*` pipeline emits
    // for closures / specialisations).  Each is a BFS root, so `blocks`
    // (the union of `iterblocks()` over each entry) covers every block
    // reachable from any entry.
    let with_predecessor: HashSet<BlockId> = graph
        .blocks
        .iter()
        .flat_map(|b| b.exits.iter().map(|e| e.target))
        .collect();
    let start_blocks: HashSet<BlockId> = std::iter::once(start)
        .chain(
            graph
                .blocks
                .iter()
                .map(|b| b.id)
                .filter(|id| !with_predecessor.contains(id)),
        )
        .collect();

    // BFS reachability mirrors `flowspace/model.py:66 iterblocks()`,
    // unioned across `start_blocks`:
    //
    //     def iterblocks(self):
    //         block = self.startblock
    //         yield block
    //         seen = {block: True}
    //         stack = list(block.exits[::-1])
    //         while stack:
    //             block = stack.pop().target
    //             if block not in seen:
    //                 yield block
    //                 seen[block] = True
    //                 stack += block.exits[::-1]
    let reachable: HashSet<BlockId> = {
        let mut seen: HashSet<BlockId> = HashSet::new();
        let mut stack: Vec<BlockId> = Vec::new();
        for &sb in &start_blocks {
            if seen.insert(sb) {
                if let Some(&i) = block_index.get(&sb) {
                    stack.extend(graph.blocks[i].exits.iter().rev().map(|e| e.target));
                }
            }
        }
        while let Some(target) = stack.pop() {
            if seen.insert(target) {
                if let Some(&i) = block_index.get(&target) {
                    stack.extend(graph.blocks[i].exits.iter().rev().map(|e| e.target));
                }
            }
        }
        seen
    };

    // Step 1 + 2: gather initial `read_vars` (op operands of non-pure
    // ops + exitswitches + terminal-block inputargs) and
    // `dependencies` (target inputarg ← link arg, plus pure-op
    // operands routed via `op.result`).
    //
    // `simplify.py:435-436 canremove`:
    //
    //     def canremove(op, block):
    //         return op.opname in CanRemove and op is not block.raising_op
    //
    // `block.raising_op` (`flowspace/model.py:218-221`) is
    // `block.operations[-1]` whenever `block.canraise` (i.e.
    // `block.exitswitch is c_last_exception`).  Pure-classified
    // ops parked there as overflow / zero-divide checks
    // (`int_add_ovf`, `int_floordiv_zer`, ...) MUST NOT be DCE'd
    // even if their result vid is unread — the raise side-effect
    // is observable.
    let mut read_vars: HashSet<ValueId> = HashSet::new();
    let mut dependencies: HashMap<ValueId, Vec<ValueId>> = HashMap::new();
    for block in &graph.blocks {
        if !reachable.contains(&block.id) {
            continue;
        }
        let raising_op_idx = if block.canraise() && !block.operations.is_empty() {
            Some(block.operations.len() - 1)
        } else {
            None
        };
        for (i, op) in block.operations.iter().enumerate() {
            let operands = op_value_refs(&op.kind);
            // `simplify.py:441-445`:
            //   if not canremove(op, block):    read_vars.update(args)
            //   else:                           dependencies[result] += args
            let removable = is_pure_op(&op.kind) && Some(i) != raising_op_idx;
            if let Some(result) = op.result
                && removable
            {
                dependencies.entry(result).or_default().extend(operands);
            } else {
                for vid in operands {
                    read_vars.insert(vid);
                }
            }
        }
        if let Some(ExitSwitch::Value(vid)) = &block.exitswitch {
            read_vars.insert(*vid);
        }
        // `simplify.py:459-462`: terminal blocks (no exits)
        // implicitly use every inputarg.
        if block.exits.is_empty() {
            for &iarg in &block.inputargs {
                read_vars.insert(iarg);
            }
        }
        for link in &block.exits {
            // `simplify.py:512-513` len-equality is asserted at the
            // exits-walk; the dependency-routing pass exercises the
            // same invariant by zipping `link.args` with
            // `link.target.inputargs`.
            let &target_idx = block_index
                .get(&link.target)
                .expect("simplify.py:512 — link.target must be a graph block");
            let target_block = &graph.blocks[target_idx];
            assert_eq!(
                link.args.len(),
                target_block.inputargs.len(),
                "simplify.py:513 — len(link.args) == len(link.target.inputargs)",
            );
            for (arg, &target_iarg) in link.args.iter().zip(target_block.inputargs.iter()) {
                if let LinkArg::Value(arg_vid) = arg {
                    dependencies.entry(target_iarg).or_default().push(*arg_vid);
                }
            }
        }
    }
    // Step 3: pin every `start_blocks` inputarg (multi-graph branch
    // `simplify.py:431-433`, with `graph.startblock` and orphan-entry
    // blocks treated as additional starts per the function-level doc
    // comment).
    for &sb in &start_blocks {
        if let Some(&i) = block_index.get(&sb) {
            for &iarg in &graph.blocks[i].inputargs {
                read_vars.insert(iarg);
            }
        }
    }

    // Step 4: backward flow.
    // `simplify.py:471-479 flow_read_var_backward`.
    let mut pending: Vec<ValueId> = read_vars.iter().copied().collect();
    while let Some(var) = pending.pop() {
        if let Some(deps) = dependencies.get(&var) {
            for &dep in deps {
                if read_vars.insert(dep) {
                    pending.push(dep);
                }
            }
        }
    }

    // Step 5: drop every pure op whose result is dead.
    // `simplify.py:484-488 if op.result not in read_vars: if
    // canremove(op, block): del block.operations[i]`.
    // Inputarg-shaped Input ops keep their result vid in `read_vars`
    // via Step 1+3+dependency-routing, so the live ones survive
    // unconditionally; dead ones get removed alongside their
    // matching inputarg trim in Step 7.  Naked Input ops (the
    // legacy fallback at `front/ast.rs` for cross-block reads not
    // yet routed through `lazy_install_local_at_current_block`) do
    // NOT have inputarg pinning, so the same `result not in
    // read_vars` predicate retires them — `simplify.py` itself has
    // no standalone Input op shape, but the line-by-line port
    // sweeps them under the same canremove rule.
    //
    // The reverse `for i in range(len-1, -1, -1)` walk
    // (`simplify.py:484`) is preserved: we resolve `raising_op_idx`
    // before the loop and check `Some(i) != raising_op_idx` so a
    // pure op that happens to be the block's raising_op
    // (`canremove`'s `op is not block.raising_op` clause,
    // `simplify.py:436`) survives even when its result is unread.
    for block in &mut graph.blocks {
        if !reachable.contains(&block.id) {
            continue;
        }
        let raising_op_idx = if block.canraise() && !block.operations.is_empty() {
            Some(block.operations.len() - 1)
        } else {
            None
        };
        for i in (0..block.operations.len()).rev() {
            let op = &block.operations[i];
            let dead = match op.result {
                Some(r) => !read_vars.contains(&r),
                None => false,
            };
            if dead && is_pure_op(&op.kind) && Some(i) != raising_op_idx {
                block.operations.remove(i);
            }
        }
    }

    // Step 6: trim Link.args at indices whose target inputarg is dead.
    // `simplify.py:512-516`.  Walk in reverse so removals don't shift
    // surviving indices.
    for block_idx in 0..graph.blocks.len() {
        if !reachable.contains(&graph.blocks[block_idx].id) {
            continue;
        }
        let exits_len = graph.blocks[block_idx].exits.len();
        for exit_idx in 0..exits_len {
            let target = graph.blocks[block_idx].exits[exit_idx].target;
            let target_iargs = {
                let &i = block_index
                    .get(&target)
                    .expect("simplify.py:512 — link.target must be a graph block");
                graph.blocks[i].inputargs.clone()
            };
            assert_eq!(
                graph.blocks[block_idx].exits[exit_idx].args.len(),
                target_iargs.len(),
                "simplify.py:513 — len(link.args) == len(link.target.inputargs)",
            );
            for i in (0..target_iargs.len()).rev() {
                let target_vid = target_iargs[i];
                if !read_vars.contains(&target_vid) {
                    graph.blocks[block_idx].exits[exit_idx].args.remove(i);
                }
            }
        }
    }

    // Step 7: drop dead inputargs + their matching `OpKind::Input` ops.
    // `simplify.py:520-524`.  Walk every reachable block; startblock is
    // already pinned in Step 3, return / except blocks in Step 1.
    for block_idx in 0..graph.blocks.len() {
        let block_id = graph.blocks[block_idx].id;
        if !reachable.contains(&block_id) || block_id == return_block || block_id == except_block {
            continue;
        }
        let inputargs = graph.blocks[block_idx].inputargs.clone();
        for i in (0..inputargs.len()).rev() {
            let vid = inputargs[i];
            if read_vars.contains(&vid) {
                continue;
            }
            graph.blocks[block_idx].inputargs.remove(i);
            if let Some(op_idx) = graph.blocks[block_idx]
                .operations
                .iter()
                .position(|op| matches!(op.kind, OpKind::Input { .. }) && op.result == Some(vid))
            {
                graph.blocks[block_idx].operations.remove(op_idx);
            }
        }
    }
}

impl Block {
    pub fn canraise(&self) -> bool {
        matches!(self.exitswitch, Some(ExitSwitch::LastException))
    }

    /// RPython `flowspace/model.py:247 closeblock` / `:250 recloseblock`
    /// mark a block's exits tuple as populated.  Pyre mirrors the
    /// "has this block been closed?" predicate by checking that either
    /// `exits` has at least one `Link` or `exitswitch` is set.
    /// During graph construction, an unclosed block has
    /// `exits=[]`, `exitswitch=None` — the upstream equivalent of
    /// `type(block.exits) is list` pre-`closeblock`.
    pub fn is_closed(&self) -> bool {
        !self.exits.is_empty() || self.exitswitch.is_some()
    }

    /// Complement of `is_closed` — true if the front-end has not yet
    /// stamped a terminator / exits onto the block.  Used to gate
    /// fall-through code that adds the block's own exit.
    pub fn is_open(&self) -> bool {
        !self.is_closed()
    }
}

/// RPython `flowspace/model.py:238-244 renamevariables` applies a
/// value renaming to `inputargs` / `operations` / `exitswitch` /
/// `link.args`.  Pyre threads the renamer out-of-band here so callers
/// can reshape both the exitswitch variable and every exit link in one
/// call.
pub fn remap_control_flow_metadata<FValue, FBlock>(
    exitswitch: &Option<ExitSwitch>,
    exits: &[Link],
    remap_value: FValue,
    remap_block: FBlock,
) -> (Option<ExitSwitch>, Vec<Link>)
where
    FValue: Fn(ValueId) -> ValueId,
    FBlock: Fn(BlockId) -> BlockId,
{
    let exitswitch = exitswitch.as_ref().map(|switch| match switch {
        ExitSwitch::Value(value) => ExitSwitch::Value(remap_value(*value)),
        ExitSwitch::LastException => ExitSwitch::LastException,
    });
    let exits = exits
        .iter()
        .map(|link| Link {
            args: link
                .args
                .iter()
                .map(|arg| match arg {
                    LinkArg::Value(value) => LinkArg::Value(remap_value(*value)),
                    LinkArg::Const(value) => LinkArg::Const(value.clone()),
                })
                .collect(),
            target: remap_block(link.target),
            exitcase: link.exitcase.clone(),
            prevblock: link.prevblock.map(&remap_block),
            llexitcase: link.llexitcase.clone(),
            last_exception: link.last_exception.as_ref().map(|arg| match arg {
                LinkArg::Value(value) => LinkArg::Value(remap_value(*value)),
                LinkArg::Const(value) => LinkArg::Const(value.clone()),
            }),
            last_exc_value: link.last_exc_value.as_ref().map(|arg| match arg {
                LinkArg::Value(value) => LinkArg::Value(remap_value(*value)),
                LinkArg::Const(value) => LinkArg::Const(value.clone()),
            }),
        })
        .collect();
    (exitswitch, exits)
}

/// `Variable.concretetype` analogue collapsed to the four kinds the
/// jit_codewriter needs.
///
/// RPython stores `.concretetype` inline on each `Variable` after
/// `RPythonTyper.specialize()` rewrites the graph; pyre's
/// [`ValueId`] is an opaque index, so the per-value attribute lives
/// in [`FunctionGraph::value_types`] (mirroring upstream's
/// inline-on-Variable shape) instead of a separate side-table.
/// `Unknown` is the pre-rtyper sentinel — `Variable` analogue
/// before the rtyper ran.
///
/// The variants line up 1:1 with [`getkind`]'s output strings
/// (`rpython/jit/metainterp/history.py:45-71`):
///
/// - [`Self::Signed`] = `"int"`
/// - [`Self::GcRef`]  = `"ref"`
/// - [`Self::Float`]  = `"float"`
/// - [`Self::Void`]   = `"void"`
///
/// [`Self::Unknown`] is a pyre-only sentinel for the
/// pre-`setconcretetype` window where reading `var.concretetype`
/// upstream would `AttributeError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConcreteType {
    /// Signed integer (RPython `Signed` / i64).
    Signed,
    /// GC reference (RPython `Ptr(GcStruct)`).
    GcRef,
    /// Float (RPython `Float` / f64).
    Float,
    /// Void (RPython `Void`).
    Void,
    /// Unknown / unresolved.
    Unknown,
}

/// `rpython/jit/metainterp/history.py:45-71 getkind(TYPE, ...)`.
///
/// Direct line-by-line port — the canonical
/// `LowLevelType → 'int' / 'ref' / 'float' / 'void'` projection that
/// every JIT codewriter / metainterp / regalloc consumer reads kind
/// from.  Pyre returns the [`ConcreteType`] enum form (same kind
/// space).  The three `supports_*` parameters default to `true`
/// just like the JIT uses — `metainterp_sd` flips them off only
/// for backends without longlong / singlefloat / float support.
///
/// ```py
/// def getkind(TYPE, supports_floats=True,
///                   supports_longlong=True,
///                   supports_singlefloats=True):
///     if TYPE is lltype.Void:
///         return "void"
///     elif isinstance(TYPE, lltype.Primitive):
///         if TYPE is lltype.Float and supports_floats:
///             return 'float'
///         if TYPE is lltype.SingleFloat and supports_singlefloats:
///             return 'int'     # singlefloats are stored in an int
///         if TYPE in (lltype.Float, lltype.SingleFloat):
///             raise NotImplementedError("type %s not supported" % TYPE)
///         if (TYPE != llmemory.Address and
///             rffi.sizeof(TYPE) > rffi.sizeof(lltype.Signed)):
///             if supports_longlong and TYPE is not lltype.LongFloat:
///                 assert rffi.sizeof(TYPE) == 8
///                 return 'float'
///             raise NotImplementedError("type %s is too large" % TYPE)
///         return "int"
///     elif isinstance(TYPE, lltype.Ptr):
///         if TYPE.TO._gckind == 'raw':
///             return "int"
///         else:
///             return "ref"
///     else:
///         raise NotImplementedError("type %s not supported" % TYPE)
/// ```
pub fn getkind(ty: &crate::translator::rtyper::lltypesystem::lltype::LowLevelType) -> ConcreteType {
    use crate::translator::rtyper::lltypesystem::lltype::{GcKind, LowLevelType, PtrTarget};
    match ty {
        // `if TYPE is lltype.Void: return "void"`
        LowLevelType::Void => ConcreteType::Void,
        // `elif isinstance(TYPE, lltype.Primitive): ...`
        LowLevelType::Float => ConcreteType::Float,
        // `if TYPE is lltype.SingleFloat and supports_singlefloats: return 'int'`
        // (singlefloats stored in an int register).
        LowLevelType::SingleFloat => ConcreteType::Signed,
        // `if rffi.sizeof(TYPE) > rffi.sizeof(lltype.Signed): ...
        //   if supports_longlong and TYPE is not lltype.LongFloat: return 'float'`
        // — the (UnsignedLongLongLong, SignedLongLongLong) 16-byte
        // variants raise NotImplementedError upstream; pyre treats
        // them as `Float` slots until the long-long backend lands.
        LowLevelType::SignedLongLong | LowLevelType::UnsignedLongLong => ConcreteType::Float,
        LowLevelType::LongFloat => ConcreteType::Float,
        // Other Primitives → `"int"`.  Includes Signed, Unsigned,
        // Bool, Char, UniChar, Address.  RPython's
        // `rffi.sizeof(TYPE) > rffi.sizeof(lltype.Signed)` check
        // only fires for the 16-byte longlonglong primitives below
        // — every other primitive fits in an Int register.
        LowLevelType::Signed
        | LowLevelType::Unsigned
        | LowLevelType::Bool
        | LowLevelType::Char
        | LowLevelType::UniChar
        | LowLevelType::Address => ConcreteType::Signed,
        // `if rffi.sizeof(TYPE) > rffi.sizeof(lltype.Signed) and not supports_longlong:
        //      raise NotImplementedError("type %s is too large" % TYPE)`
        // — the 16-byte longlonglong variants surface as panics
        // upstream too.  Keep parity by panicking pyre-side until
        // the JIT supports them; production paths never reach this.
        LowLevelType::SignedLongLongLong | LowLevelType::UnsignedLongLongLong => {
            panic!("getkind: type {ty:?} is too large (longlonglong unsupported)")
        }
        // `elif isinstance(TYPE, lltype.Ptr): ...`
        LowLevelType::Ptr(ptr) => match ptr_gckind(&ptr.TO) {
            GcKind::Raw => ConcreteType::Signed,
            GcKind::Gc | GcKind::Prebuilt => ConcreteType::GcRef,
        },
        // RPython `lltype.InteriorPtr` is treated as a pointer — the
        // codewriter never sees it as an operand top-level type, so
        // mirror the `Ptr` arm conservatively.
        LowLevelType::InteriorPtr(_) => ConcreteType::GcRef,
        // RPython `Func`/`Struct`/`Array`/`FixedSizeArray`/`Opaque`/
        // `ForwardReference` are not valid `concretetype` values for
        // a Variable — they only appear as `TO` of a `Ptr`.  Reaching
        // this arm means the rtyper handed the codewriter a
        // non-pointer aggregate type, which would `NotImplementedError`
        // upstream.  Panic with the same message shape so the upstream
        // bug surfaces here.
        LowLevelType::Func(_)
        | LowLevelType::Struct(_)
        | LowLevelType::Array(_)
        | LowLevelType::FixedSizeArray(_)
        | LowLevelType::Opaque(_)
        | LowLevelType::ForwardReference(_) => {
            panic!("getkind: type {ty:?} not supported as concretetype")
        }
    }
}

/// Helper for [`getkind`] — extract `_gckind` from a `Ptr.TO`
/// payload.  RPython exposes `_gckind` as an attribute on every
/// pointee (`StructType._gckind`, `ArrayType._gckind`, etc.) so the
/// `TYPE.TO._gckind == 'raw'` test in `getkind` works directly;
/// pyre's `PtrTarget` enum surface needs a small dispatch.
fn ptr_gckind(
    target: &crate::translator::rtyper::lltypesystem::lltype::PtrTarget,
) -> crate::translator::rtyper::lltypesystem::lltype::GcKind {
    use crate::translator::rtyper::lltypesystem::lltype::PtrTarget;
    match target {
        PtrTarget::Struct(s) => s._gckind,
        PtrTarget::Array(a) => a._gckind,
        PtrTarget::FixedSizeArray(_) => {
            crate::translator::rtyper::lltypesystem::lltype::GcKind::Raw
        }
        PtrTarget::Opaque(o) => o._gckind,
        PtrTarget::ForwardReference(f) => f._gckind,
        PtrTarget::Func(_) => crate::translator::rtyper::lltypesystem::lltype::GcKind::Raw,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionGraph {
    pub name: String,
    pub startblock: BlockId,
    /// RPython `flowspace/model.py:17-18 FunctionGraph.returnblock` —
    /// `Block([return_var])` with `operations=()` and `exits=()`.
    /// Blocks returning a value route to it via
    /// `Link([value], returnblock)` held in `Block.exits`.
    pub returnblock: BlockId,
    /// RPython `FunctionGraph.exceptblock` — `Block([etype, evalue])`.
    pub exceptblock: BlockId,
    pub blocks: Vec<Block>,
    pub notes: Vec<String>,
    next_value: usize,
    /// Per-value [`ConcreteType`] indexed by [`ValueId`] — the
    /// pyre analogue of RPython's `Variable.concretetype` inline
    /// field.  `value_types[v.0]` is the kind for `ValueId(v)`,
    /// kept in lockstep with the [`Self::next_value`] allocator so
    /// every live `ValueId` always has a matching slot
    /// (`ConcreteType::Unknown` until the rtyper writes the real
    /// kind).  Replaces the previous `TypeResolutionState` side
    /// table (`AGENTS.md §2`: type information must live on the
    /// "box itself", not in a parallel `HashMap`).
    pub value_types: Vec<ConcreteType>,
    /// Variable names for debugging (RPython Variable._name).
    pub value_names: std::collections::HashMap<ValueId, String>,
    /// Slice Z2.5.A scaffolding — `flowspace::model::Variable.id` →
    /// `ValueId` bridge for the gap between upstream's polymorphic
    /// `Hlvalue` mergeable cells (Variable | Constant | FlowSignal) and
    /// pyre's `link.args: Vec<ValueId>` representation.  Populated lazily
    /// at link-arg materialization sites (the consumer is
    /// `FrameState::getoutputargs` stack / exception walks); ValueIds are
    /// minted from `alloc_value()` so they live in the same allocator
    /// space as IR operand ids and never collide.
    ///
    /// RPython has no counterpart — `framestate.py:92-99 getoutputargs`
    /// pushes the polymorphic `Hlvalue` cell directly into `Link.args`,
    /// no bridge needed because the link domain IS `Hlvalue`.  This map
    /// is the documented PRE-EXISTING-ADAPTATION for pyre's pre-Z2.5
    /// `Vec<ValueId>` link-arg shape; once the Z2.5 epic (slices A–D)
    /// completes, the bridge becomes the canonical narrow waist between
    /// the flowspace-shaped FrameState and pyre's IR.
    pub(crate) variable_to_vid: std::collections::HashMap<u64, ValueId>,
    /// The source-level return type string (Rust `syn::ReturnType`
    /// rendered through `qualified_full_type_string`). RPython
    /// equivalent: `funcptr._obj.TO.RESULT` on the
    /// `lltype.FuncType(ARGS, RESULT)` — every graph carries its result
    /// type intrinsically as `op.result.concretetype` of the
    /// returnblock's input variable.  Pyre stores the source string and
    /// projects to `Type` via `return_type_string_to_value_type` so the
    /// JIT codewriter's signature validator (`call.rs:3502/3555`) can
    /// read `FUNC.RESULT` directly off the callee graph without
    /// consulting the `CallControl::return_types` side-table.
    ///
    /// `None` for synthetic test-fixture graphs constructed via
    /// `FunctionGraph::new("name")`; production paths populate via
    /// `with_return_type(rt)` after construction (parse.rs +
    /// lib.rs:430-512 trait-method + free-fn registration).
    pub return_type: Option<String>,
}

impl FunctionGraph {
    pub fn new(name: impl Into<String>) -> Self {
        let entry = BlockId(0);
        let returnblock = BlockId(1);
        let exceptblock = BlockId(2);
        let return_value = ValueId(0);
        let last_exception = ValueId(1);
        let last_exc_value = ValueId(2);
        Self {
            name: name.into(),
            startblock: entry,
            returnblock,
            exceptblock,
            // RPython `flowspace/model.py:14-25 FunctionGraph.__init__`:
            //   startblock created empty; returnblock = Block([return_var]);
            //   exceptblock = Block([Variable('etype'), Variable('evalue')]).
            // Final blocks have `operations=()` and `exits=()`; fall-through
            // for the startblock is likewise `exits=[]` until the front-end
            // closes it.
            blocks: vec![
                Block {
                    id: entry,
                    inputargs: Vec::new(),
                    operations: Vec::new(),
                    exitswitch: None,
                    exits: Vec::new(),
                    framestate: None,
                    dead: false,
                },
                Block {
                    id: returnblock,
                    inputargs: vec![return_value],
                    operations: Vec::new(),
                    exitswitch: None,
                    exits: Vec::new(),
                    framestate: None,
                    dead: false,
                },
                Block {
                    id: exceptblock,
                    inputargs: vec![last_exception, last_exc_value],
                    operations: Vec::new(),
                    exitswitch: None,
                    exits: Vec::new(),
                    framestate: None,
                    dead: false,
                },
            ],
            notes: Vec::new(),
            next_value: 3,
            // Canonical inputargs (returnblock inputarg, exceptblock
            // etype, exceptblock evalue) start as `Unknown`.  The
            // rtyper / `regalloc::augment_value_kinds_with_canonical_exceptblock`
            // populates them with `(_, Signed, GcRef)` later in the
            // pipeline; until then `kind_color_of` falls through to
            // its bare regalloc-class scan path, matching the
            // pre-rtyper Variable-with-no-`concretetype` shape.
            value_types: vec![
                ConcreteType::Unknown,
                ConcreteType::Unknown,
                ConcreteType::Unknown,
            ],
            value_names: std::collections::HashMap::new(),
            variable_to_vid: std::collections::HashMap::new(),
            return_type: None,
        }
    }

    /// Builder-style setter for `return_type`. Used by production
    /// registration paths (parse.rs / lib.rs) where the source-level
    /// type is available; test fixtures that construct via
    /// `FunctionGraph::new("name")` may skip and leave `None`. The JIT
    /// codewriter signature validator (`call.rs:3502/3555`) falls back
    /// to `CallControl::return_types` for `None`-carrying graphs.
    pub fn with_return_type(mut self, rt: impl Into<String>) -> Self {
        self.return_type = Some(rt.into());
        self
    }

    /// Return the canonical exception block and its `(etype, evalue)`
    /// inputargs.
    ///
    /// RPython parity: `flowspace/model.py:21-25` `exceptblock` has
    /// two inputargs, `(etype, evalue)`, and exists eagerly on every
    /// graph.
    pub fn exceptblock_args(&self) -> (BlockId, ValueId, ValueId) {
        let args = &self.block(self.exceptblock).inputargs;
        (self.exceptblock, args[0], args[1])
    }

    /// Return the canonical return block and its single inputarg.
    ///
    /// RPython parity: `FunctionGraph.getreturnvar()` reads
    /// `graph.returnblock.inputargs[0]`.
    pub fn returnblock_arg(&self) -> (BlockId, ValueId) {
        let args = &self.block(self.returnblock).inputargs;
        (self.returnblock, args[0])
    }

    pub fn create_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len());
        self.blocks.push(Block {
            id,
            inputargs: Vec::new(),
            operations: Vec::new(),
            exitswitch: None,
            exits: Vec::new(),
            framestate: None,
            dead: false,
        });
        id
    }

    /// Create a block with explicit inputargs (Phi nodes).
    pub fn create_block_with_args(&mut self, num_args: usize) -> (BlockId, Vec<ValueId>) {
        let id = BlockId(self.blocks.len());
        let args: Vec<ValueId> = (0..num_args).map(|_| self.alloc_value()).collect();
        self.blocks.push(Block {
            id,
            inputargs: args.clone(),
            operations: Vec::new(),
            exitswitch: None,
            exits: Vec::new(),
            framestate: None,
            dead: false,
        });
        (id, args)
    }

    pub fn alloc_value(&mut self) -> ValueId {
        self.alloc_value_with_type(ConcreteType::Unknown)
    }

    /// Allocate a fresh [`ValueId`] with its [`ConcreteType`]
    /// stamped at construction — pyre's analogue of upstream
    /// `Variable(concretetype=...)` (RPython
    /// `flowspace/model.py:Variable.__init__`).  Front-ends and the
    /// rtyper should prefer this entry point so the per-value
    /// kind never has to be back-filled via a second pass.
    pub fn alloc_value_with_type(&mut self, ty: ConcreteType) -> ValueId {
        let id = ValueId(self.next_value);
        self.next_value += 1;
        // Keep `value_types` in lockstep with `next_value`.  Any
        // gap means a downstream `concretetype(v)` would silently
        // see `Unknown` — keep the invariant explicit.
        debug_assert_eq!(
            self.value_types.len(),
            id.0,
            "value_types length must equal the next ValueId index ({} != {})",
            self.value_types.len(),
            id.0,
        );
        self.value_types.push(ty);
        id
    }

    /// Slice Z2.5.A — get-or-mint the `ValueId` bridge for an upstream
    /// `Variable` cell (key = `Variable.id`).  Idempotent: repeated calls
    /// with the same Variable return the same ValueId; distinct
    /// Variables get distinct ValueIds via `alloc_value()`.
    ///
    /// Consumers populate the bridge at the moment a `Variable`
    /// originating from `FrameState.stack` or `last_exception` needs
    /// to cross a `Link.args` boundary — see
    /// `FrameState::getoutputargs` (Z2.5.B+).  Today's AST frontend
    /// keeps both projections empty, so this helper has no production
    /// caller yet; the Z2.5.B slice wires it in.
    pub fn bridge_variable(&mut self, v: &crate::flowspace::model::Variable) -> ValueId {
        if let Some(&vid) = self.variable_to_vid.get(&v.id()) {
            return vid;
        }
        let vid = self.alloc_value();
        self.variable_to_vid.insert(v.id(), vid);
        vid
    }

    /// `Variable.concretetype` getter — every live [`ValueId`] has
    /// a slot in [`Self::value_types`].
    pub fn concretetype(&self, v: ValueId) -> &ConcreteType {
        self.value_types
            .get(v.0)
            .unwrap_or(&ConcreteType::Unknown)
    }

    /// Late-stamp [`ValueId`] kind (e.g. after the rtyper resolved
    /// a previously-`Unknown` slot).  Asserts the slot exists so
    /// out-of-range writes surface immediately.
    pub fn set_concretetype(&mut self, v: ValueId, ty: ConcreteType) {
        if v.0 >= self.value_types.len() {
            // Grow the type table if a value was minted via
            // `set_next_value` (jtransform synthesises ValueIds
            // outside the graph for a brief window).  Filling the
            // gap with `Unknown` keeps the invariant
            // `value_types.len() == next_value`.
            self.value_types
                .resize(v.0 + 1, ConcreteType::Unknown);
        }
        self.value_types[v.0] = ty;
    }

    /// Read-only view of the ValueId allocator cursor.  Used by passes
    /// that need to mint fresh ValueIds outside the graph (e.g.
    /// `Transformer::allocate_synthetic_value` in `jtransform.rs`).
    pub fn next_value(&self) -> usize {
        self.next_value
    }

    /// Re-seat the ValueId allocator cursor.  Must be called after a
    /// pass that synthesized values outside the graph so subsequent
    /// `alloc_value()` calls do not collide.
    pub fn set_next_value(&mut self, next: usize) {
        debug_assert!(
            next >= self.next_value,
            "set_next_value must not walk the cursor backward: {} -> {}",
            self.next_value,
            next,
        );
        // Grow the per-value kind table to match the new cursor so
        // the `value_types.len() == next_value` invariant holds.
        // The synthesised ValueIds default to `Unknown` and the
        // rtyper / jtransform should follow up with
        // `set_concretetype` once they resolve the kind.
        if next > self.value_types.len() {
            self.value_types
                .resize(next, ConcreteType::Unknown);
        }
        self.next_value = next;
    }

    pub fn push_op(&mut self, block: BlockId, kind: OpKind, has_result: bool) -> Option<ValueId> {
        let result = has_result.then(|| self.alloc_value());
        self.blocks[block.0]
            .operations
            .push(SpaceOperation { result, kind });
        result
    }

    /// Push an op with a caller-supplied `result` ValueId.  Used when
    /// the vid was allocated upstream (e.g. `FrameState::union`
    /// pre-allocates phi vids) and the op now needs to be emitted with
    /// the same vid.
    pub fn push_op_with_result(&mut self, block: BlockId, kind: OpKind, result: ValueId) {
        self.blocks[block.0].operations.push(SpaceOperation {
            result: Some(result),
            kind,
        });
    }

    /// RPython `flowspace/model.py:250 recloseblock(*exits)` — stamp
    /// `link.prevblock` on each exit and install them as the block's
    /// exits, overwriting any previous contents.  The `exitswitch`
    /// field is passed alongside so callers who set a bool branch or
    /// can-raise shape keep both halves of the canonical CFG surface
    /// updated in a single call (pyre-side ergonomics; upstream writes
    /// `block.exitswitch = ...` before `closeblock`).
    pub fn set_control_flow_metadata(
        &mut self,
        block: BlockId,
        exitswitch: Option<ExitSwitch>,
        exits: Vec<Link>,
    ) {
        let block_ref = &mut self.blocks[block.0];
        block_ref.exitswitch = exitswitch;
        block_ref.exits = exits
            .into_iter()
            .map(|link| link.with_prevblock(block))
            .collect();
    }

    /// RPython `flowspace/model.py:250 recloseblock(*exits)` — stamp
    /// `link.prevblock` on each exit and install them as the block's
    /// exits, overwriting any previous contents.  Like upstream, this
    /// does not touch `exitswitch`; callers that want to change the
    /// branch/raise discriminator must set it separately before
    /// `closeblock`/`recloseblock`.
    pub fn recloseblock(&mut self, block: BlockId, exits: Vec<Link>) {
        self.blocks[block.0].exits = exits
            .into_iter()
            .map(|link| link.with_prevblock(block))
            .collect();
    }

    /// RPython `flowspace/model.py:246 closeblock(*exits)` —
    /// `assert self.exits == [], "block already closed"` before
    /// delegating to `recloseblock`.  Keep the invariant as a regular
    /// assert, not `debug_assert!`, so release builds match upstream's
    /// fail-fast behavior.
    pub fn closeblock(&mut self, block: BlockId, exits: Vec<Link>) {
        assert!(
            self.blocks[block.0].exits.is_empty(),
            "block {:?} already closed",
            block
        );
        self.recloseblock(block, exits);
    }

    /// Shorthand for the single-exit fall-through shape — one Link to
    /// `target` carrying `args`, `exitswitch = None`.  Upstream
    /// equivalent: `block.closeblock(Link(args, target))`
    /// (`flowspace/model.py:304`).
    pub fn set_goto(&mut self, block: BlockId, target: BlockId, args: Vec<ValueId>) {
        self.set_control_flow_metadata(block, None, vec![Link::new(args, target, None)]);
    }

    /// Shorthand for the boolean-branch shape — two Links with
    /// `Bool(false)` / `Bool(true)` exitcases, `exitswitch =
    /// ExitSwitch::Value(cond)`.  Upstream equivalent:
    /// `block.exitswitch = cond;
    ///  block.closeblock(Link(false_args, if_false, False),
    ///                   Link(true_args,  if_true,  True))`
    /// (`flowspace/model.py:175-180` + `:304`).
    pub fn set_branch(
        &mut self,
        block: BlockId,
        cond: ValueId,
        if_true: BlockId,
        true_args: Vec<ValueId>,
        if_false: BlockId,
        false_args: Vec<ValueId>,
    ) {
        self.set_control_flow_metadata(
            block,
            Some(ExitSwitch::Value(cond)),
            vec![
                Link::new(false_args, if_false, Some(ExitCase::Bool(false)))
                    .with_llexitcase_from_exitcase(),
                Link::new(true_args, if_true, Some(ExitCase::Bool(true)))
                    .with_llexitcase_from_exitcase(),
            ],
        );
    }

    /// Route a return through the graph's canonical `returnblock`.
    ///
    /// RPython `flowcontext.py` return handling produces a fresh
    /// prevblock-side Variable (Void Variable for `return None`), then
    /// builds a Link carrying that value into the returnblock's
    /// inputargs.  pyre's codewriter adaptation mirrors that shape: a
    /// `None` `value` allocates a fresh prevblock-side ValueId whose
    /// kind defaults to Void (no regalloc color, no emitted move), so
    /// `Link.args` is always a prevblock value per upstream's
    /// `flowspace/model.py:114` invariant.
    pub fn set_return(&mut self, block: BlockId, value: Option<ValueId>) {
        let (returnblock, _) = self.returnblock_arg();
        let value = value.unwrap_or_else(|| self.alloc_value());
        self.set_goto(block, returnblock, vec![value]);
    }

    /// Route `block` to the graph's canonical `exceptblock` — the
    /// upstream-shaped exit for an unrecoverable exception.
    ///
    /// RPython `flowspace/model.py:21-25` declares
    /// `exceptblock = Block([Variable('etype'), Variable('evalue')])` as
    /// the single raise destination per graph.  Predecessor blocks route
    /// to it via `Link(args=[etype, evalue], target=exceptblock)` held in
    /// `Block.exits` — there is no upstream terminator variant that
    /// flags "this block raises".
    ///
    /// Emits the same CFG shape as upstream: a Link to
    /// `graph.exceptblock` with two fresh prevblock-side ValueIds
    /// standing in for the `(etype, evalue)` pair.  The `_reason`
    /// string is retained for optional GraphTransformNote annotations
    /// (see `jtransform.rs::rewrite_graph`'s abort note); pass `""`
    /// when not applicable.
    ///
    /// Used only where no concrete exception payload is available at
    /// the raise site — the evaluated arguments path should call
    /// `set_raise_values` instead.
    pub fn set_raise(&mut self, block: BlockId, _reason: &str) {
        let (exceptblock, _, _) = self.exceptblock_args();
        let etype = self.alloc_value();
        let evalue = self.alloc_value();
        self.set_goto(block, exceptblock, vec![etype, evalue]);
    }

    /// Terminate `block` with a Link to `exceptblock` carrying the
    /// caller-provided `(etype, evalue)` pair.
    ///
    /// RPython `flowspace/flowcontext.py:1253 Raise.nomoreblocks`:
    /// ```python
    /// raise FSException(self.w_type, self.w_value)
    /// ```
    /// The two `W_*` values flow from the preceding `RAISE_VARARGS`
    /// at `flowspace/flowcontext.py:638-656`, where `popvalue()`s
    /// provide the exception type / value already on the stack.
    /// This API lets pyre's macro lowering thread the evaluated
    /// ValueIds into the exceptblock Link so the exception payload
    /// is preserved, not discarded in favor of fresh `alloc_value()`
    /// placeholders as `set_raise` does.
    pub fn set_raise_values(&mut self, block: BlockId, etype: ValueId, evalue: ValueId) {
        let (exceptblock, _, _) = self.exceptblock_args();
        self.set_goto(block, exceptblock, vec![etype, evalue]);
    }

    pub fn block(&self, block: BlockId) -> &Block {
        &self.blocks[block.0]
    }

    /// Name a value (RPython Variable._name).
    pub fn name_value(&mut self, id: ValueId, name: impl Into<String>) {
        self.value_names.insert(id, name.into());
    }

    /// Get the name of a value, if any.
    pub fn value_name(&self, id: ValueId) -> Option<&str> {
        self.value_names.get(&id).map(|s| s.as_str())
    }

    /// Whether `id` is the result of a constant-define `OpKind` —
    /// RPython's analogue is `Constant` (immutable wrapper around a
    /// Python literal).  STORE_FAST upstream
    /// (`flowspace/flowcontext.py:878-885`) only renames the rhs
    /// when it is a `Variable`; pyre's `name_value` is a ValueId-
    /// keyed metadata side table that would otherwise attach a
    /// local name to a constant define-op, breaking the 1:1
    /// structural correspondence.
    ///
    /// **Maintenance contract**: every new constant-producing
    /// `OpKind` variant must be added to the predicate below.
    /// Today the only constant-define `OpKind`s are `ConstInt` and
    /// `ConstFloat`; if pyre's frontend later adds e.g.
    /// `ConstBool` / `ConstStr` / `ConstNone` opvariants for
    /// `LOAD_CONST`-equivalents
    /// (`flowcontext.py:858-869` + `operation.py:152`), this match
    /// must extend to cover them so the STORE_FAST rename gate
    /// stays parity-faithful.
    pub fn is_constant_define_value(&self, id: ValueId) -> bool {
        self.iter_block_ops().any(|(_, op)| {
            op.result == Some(id)
                && match &op.kind {
                    // RPython `Constant(int_value)` — integer-literal
                    // define op.
                    OpKind::ConstInt(_) => true,
                    // RPython `Constant(bool_value)` (lltype.Bool).
                    OpKind::ConstBool(_) => true,
                    // RPython `Constant(float_value)`.
                    OpKind::ConstFloat(_) => true,
                    // Every other variant is computed via flowspace
                    // operators or recorder hooks; results are
                    // RPython `Variable`s.
                    _ => false,
                }
        })
    }

    pub fn block_mut(&mut self, block: BlockId) -> &mut Block {
        &mut self.blocks[block.0]
    }

    // ── RPython FunctionGraph iteration methods ──────────────────

    /// Iterate all blocks. RPython: `graph.iterblocks()`.
    pub fn iter_blocks(&self) -> impl Iterator<Item = &Block> {
        self.blocks.iter()
    }

    /// Iterate all (block, op) pairs. RPython: `graph.iterblockops()`.
    pub fn iter_block_ops(&self) -> impl Iterator<Item = (&Block, &SpaceOperation)> {
        self.blocks
            .iter()
            .flat_map(|b| b.operations.iter().map(move |op| (b, op)))
    }

    /// Get successor block IDs for a block.
    ///
    /// RPython `flowspace/model.py:66-76 FunctionGraph.iterblocks`:
    /// successor set is derived from `Block.exits` only.  Final blocks
    /// (`exits == ()`) — returnblock / exceptblock — have no successors.
    pub fn successors(&self, block: BlockId) -> Vec<BlockId> {
        self.block(block)
            .exits
            .iter()
            .map(|link| link.target)
            .collect()
    }

    /// Get predecessor block IDs for a block.
    pub fn predecessors(&self, target: BlockId) -> Vec<BlockId> {
        self.blocks
            .iter()
            .filter(|b| self.successors(b.id).contains(&target))
            .map(|b| b.id)
            .collect()
    }

    /// Count total operations across all blocks.
    pub fn num_ops(&self) -> usize {
        self.blocks.iter().map(|b| b.operations.len()).sum()
    }

    /// Check if a block is a loop header (has a back-edge predecessor).
    pub fn is_loop_header(&self, block: BlockId) -> bool {
        self.predecessors(block)
            .iter()
            .any(|&pred| pred.0 >= block.0)
    }

    /// Pretty-print the graph (RPython `graph.show()`).
    pub fn dump(&self) -> String {
        let mut out = format!(
            "=== {} ({} blocks, {} ops) ===\n",
            self.name,
            self.blocks.len(),
            self.num_ops()
        );
        for block in &self.blocks {
            let args: Vec<String> = block.inputargs.iter().map(|v| self.fmt_value(*v)).collect();
            if args.is_empty() {
                out.push_str(&format!("  Block {}:\n", block.id.0));
            } else {
                out.push_str(&format!("  Block {}({}):\n", block.id.0, args.join(", ")));
            }
            for op in &block.operations {
                let result = op
                    .result
                    .map(|v| format!("{} = ", self.fmt_value(v)))
                    .unwrap_or_default();
                out.push_str(&format!("    {}{:?}\n", result, op.kind));
            }
            // Upstream `flowspace/model.py:199 __repr__` prints the block
            // shape as "block@N with K exits[(exitswitch)]".  Mirror the
            // same summary from pyre's canonical exitswitch/exits pair.
            match &block.exitswitch {
                Some(switch) => out.push_str(&format!(
                    "    → {} exits ({:?})\n",
                    block.exits.len(),
                    switch
                )),
                None => out.push_str(&format!("    → {} exits\n", block.exits.len())),
            }
        }
        out
    }

    fn fmt_value(&self, id: ValueId) -> String {
        if let Some(name) = self.value_name(id) {
            format!("v{}:{}", id.0, name)
        } else {
            format!("v{}", id.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_allocates_values_and_blocks() {
        let mut graph = FunctionGraph::new("demo");
        let entry = graph.startblock;
        let cond = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "x".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let next = graph.create_block();
        graph.set_branch(entry, cond, next, vec![], next, vec![]);
        assert_eq!(graph.blocks.len(), 4);
        assert_eq!(graph.block(entry).operations.len(), 1);
        assert_eq!(graph.block(graph.returnblock).inputargs.len(), 1);
        assert_eq!(graph.block(graph.exceptblock).inputargs.len(), 2);
    }

    #[test]
    fn set_control_flow_metadata_stamps_prevblock() {
        let mut graph = FunctionGraph::new("demo");
        let entry = graph.startblock;
        let next = graph.create_block();
        graph.set_control_flow_metadata(entry, None, vec![Link::new(vec![], next, None)]);
        assert_eq!(graph.block(entry).exits[0].prevblock, Some(entry));
    }

    #[test]
    fn set_return_routes_non_void_returns_via_returnblock() {
        let mut graph = FunctionGraph::new("demo");
        let entry = graph.startblock;
        let value = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "x".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(value));
        // Upstream `flowspace/model.py:171-180` identifies the routed
        // return by Block.exits carrying a single Link(value, returnblock)
        // with exitswitch=None.
        let entry_block = graph.block(entry);
        assert!(entry_block.exitswitch.is_none());
        assert_eq!(entry_block.exits.len(), 1);
        assert_eq!(entry_block.exits[0].prevblock, Some(entry));
        assert_eq!(entry_block.exits[0].target, graph.returnblock);
        assert_eq!(entry_block.exits[0].args, vec![LinkArg::from(value)]);
    }

    #[test]
    fn recloseblock_preserves_existing_exitswitch() {
        let mut graph = FunctionGraph::new("demo");
        let entry = graph.startblock;
        let cond = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "cond".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let target = graph.create_block();
        graph.block_mut(entry).exitswitch = Some(ExitSwitch::Value(cond));

        graph.recloseblock(entry, vec![Link::new(vec![], target, None)]);

        assert_eq!(graph.block(entry).exitswitch, Some(ExitSwitch::Value(cond)));
        assert_eq!(graph.block(entry).exits[0].prevblock, Some(entry));
    }

    #[test]
    #[should_panic(expected = "already closed")]
    fn closeblock_panics_when_called_twice() {
        let mut graph = FunctionGraph::new("demo");
        let entry = graph.startblock;
        let first = graph.create_block();
        let second = graph.create_block();

        graph.closeblock(entry, vec![Link::new(vec![], first, None)]);
        graph.closeblock(entry, vec![Link::new(vec![], second, None)]);
    }

    #[test]
    fn framestate_union_kills_one_sided_local() {
        // RPython parity: `framestate.py:110-111` "if w1 is None or w2
        // is None: return None" — a slot present in one predecessor
        // but missing in the other is killed (dropped) from the merged
        // state.  Pyre realises this via dense positional entries:
        // the graph-wide first-bind order assigns slots [x=0,
        // self_only=1, other_only=2]; `self` has slot 1 bound but slot
        // 2 unbound, `other` has slot 2 bound but slot 1 unbound, so
        // both one-sided slots collapse to None-kill at union.
        let self_state = FrameState {
            entries: vec![Some(ValueId(0)), Some(ValueId(1)), None],
            ..Default::default()
        };
        let other_state = FrameState {
            entries: vec![Some(ValueId(2)), None, Some(ValueId(3))],
            ..Default::default()
        };
        let mut graph = FunctionGraph::new("test");
        // Reserve vid space past the test fixtures' hardcoded vids so
        // a fresh allocation for the disagreement on slot 0 picks an
        // id distinguishable from both predecessors.
        graph.set_next_value(100);
        let merged = self_state.union(&other_state, &mut graph).expect(
            "test invariant: AST frontend union is total — entries domain has no \
             UnionError, vestigial stack/exc/blocklist/next_offset (framestate.py:78)",
        );
        assert_eq!(
            merged.entries.len(),
            3,
            "merged state preserves positional length = max(len)"
        );
        let surviving =
            merged.entries[0].expect("slot 0 (x) is bound on both sides and must survive");
        // Slot 0's predecessors disagreed on `value_id` (0 vs 2), so
        // `union` allocated a fresh vid via `graph.alloc_value()`.
        assert_ne!(surviving, ValueId(0));
        assert_ne!(surviving, ValueId(2));
        assert_eq!(surviving, ValueId(100));
        // Slots 1 (`self_only`) and 2 (`other_only`) are bound on
        // exactly one side each, so both collapse to None-kill —
        // upstream `framestate.py:110-111` "if w1 or w2 is None: …
        // return None" semantics.  Positional verification (slot
        // index = name index per graph-wide first-bind order)
        // replaces the prior name-string scan.
        assert!(
            merged.entries[1].is_none() && merged.entries[2].is_none(),
            "one-sided slots must be None-killed positionally"
        );
    }

    /// Convenience helper: install an `OpKind::Input` op AND register its
    /// vid as a phi inputarg on the same block.  Mirrors what the
    /// Slice 2 eager-phi installer at union sites does.
    fn install_phi(graph: &mut FunctionGraph, block: BlockId, name: &str) -> ValueId {
        let vid = graph
            .push_op(
                block,
                OpKind::Input {
                    name: name.to_string(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.block_mut(block).inputargs.push(vid);
        vid
    }

    #[test]
    fn prune_dead_phis_drops_orphan_phi_and_link_arg() {
        // entry → merge (phi 'x' never read) → returnblock(void)
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let const_v = graph.push_op(entry, OpKind::ConstInt(42), true).unwrap();
        let merge = graph.create_block();
        graph.set_goto(entry, merge, vec![const_v]);
        install_phi(&mut graph, merge, "x");
        graph.set_return(merge, None);

        prune_dead_phis(&mut graph);

        assert!(
            graph.block(merge).inputargs.is_empty(),
            "orphan phi inputarg must be removed"
        );
        let entry_exit = &graph.block(entry).exits[0];
        assert!(
            entry_exit.args.is_empty(),
            "predecessor link arg matching the pruned phi must be removed"
        );
        let has_phi_op = graph
            .block(merge)
            .operations
            .iter()
            .any(|op| matches!(&op.kind, OpKind::Input { name, .. } if name == "x"));
        assert!(!has_phi_op, "orphan phi `OpKind::Input` must be dropped");
    }

    #[test]
    fn prune_dead_phis_keeps_phi_with_reader() {
        // entry → merge(phi 'x' read by a BinOp whose result is the
        // function return value) → returnblock(reads return value).
        // The BinOp's result is genuinely live — it flows through the
        // returnblock's terminal-inputarg pin — so backward dataflow
        // marks it `read_vars`, then propagates back through the
        // pure-op dependencies entry to phi_x, keeping phi_x alive.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let const_v = graph.push_op(entry, OpKind::ConstInt(7), true).unwrap();
        let merge = graph.create_block();
        graph.set_goto(entry, merge, vec![const_v]);
        let phi_x = install_phi(&mut graph, merge, "x");
        // BinOp whose result IS read (by `set_return`).  Backward
        // dataflow needs a live consumer of the BinOp result for the
        // pure-op-args→dependencies routing to keep phi_x alive.
        let doubled = graph
            .push_op(
                merge,
                OpKind::BinOp {
                    op: "int_add".into(),
                    lhs: phi_x,
                    rhs: phi_x,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(merge, Some(doubled));

        prune_dead_phis(&mut graph);

        assert_eq!(
            graph.block(merge).inputargs,
            vec![phi_x],
            "phi with a live downstream reader must stay"
        );
        let entry_exit = &graph.block(entry).exits[0];
        assert_eq!(
            entry_exit.args.len(),
            1,
            "predecessor link arg matching a kept phi must stay"
        );
    }

    #[test]
    fn prune_dead_phis_collapses_dead_phi_feeding_dead_pure_op() {
        // A phi whose only "reader" is a pure op whose result is itself
        // dead.  Both must collapse together — `simplify.py:441-445`'s
        // `dependencies[op.result] += op.args` for `canremove` ops
        // means the phi vid never enters `read_vars` (because the
        // pure op's result never enters either), and Step 5's blanket
        // pure-op DCE removes the orphan op so the phi can be safely
        // trimmed at Step 7.
        //
        // entry → merge(phi 'x', dead-result BinOp reading phi_x) →
        // returnblock(void).
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let const_v = graph.push_op(entry, OpKind::ConstInt(7), true).unwrap();
        let merge = graph.create_block();
        graph.set_goto(entry, merge, vec![const_v]);
        let phi_x = install_phi(&mut graph, merge, "x");
        let doubled = graph
            .push_op(
                merge,
                OpKind::BinOp {
                    op: "int_add".into(),
                    lhs: phi_x,
                    rhs: phi_x,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        // Void return — `doubled` has no live consumer.
        graph.set_return(merge, None);

        prune_dead_phis(&mut graph);

        assert!(
            graph.block(merge).inputargs.is_empty(),
            "phi feeding only a dead pure op must collapse"
        );
        let has_binop = graph
            .block(merge)
            .operations
            .iter()
            .any(|op| op.result == Some(doubled) && matches!(op.kind, OpKind::BinOp { .. }));
        assert!(!has_binop, "dead pure op must be removed alongside the phi");
        let entry_exit = &graph.block(entry).exits[0];
        assert!(
            entry_exit.args.is_empty(),
            "predecessor link arg matching the pruned phi must be removed"
        );
    }

    #[test]
    fn prune_dead_phis_collapses_chained_dead_phis() {
        // entry → merge1(phi 'x') → merge2(phi 'y' fed by phi 'x') → returnblock(void)
        // Neither 'x' nor 'y' is read.  Backward dataflow correctly
        // identifies both as dead in a single pass: 'y' is unread, so
        // its dependency `phi_x` (link arg from merge1) doesn't get
        // promoted, and 'x' itself is unread either.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let const_v = graph.push_op(entry, OpKind::ConstInt(1), true).unwrap();
        let merge1 = graph.create_block();
        graph.set_goto(entry, merge1, vec![const_v]);
        let phi_x = install_phi(&mut graph, merge1, "x");
        let merge2 = graph.create_block();
        graph.set_goto(merge1, merge2, vec![phi_x]);
        install_phi(&mut graph, merge2, "y");
        graph.set_return(merge2, None);

        prune_dead_phis(&mut graph);

        assert!(
            graph.block(merge1).inputargs.is_empty(),
            "chained dead phi 'x' must collapse after 'y' goes"
        );
        assert!(
            graph.block(merge2).inputargs.is_empty(),
            "leaf dead phi 'y' must be pruned"
        );
        assert!(
            graph.block(entry).exits[0].args.is_empty(),
            "entry → merge1 link arg dropped after 'x' pruned"
        );
        assert!(
            graph.block(merge1).exits[0].args.is_empty(),
            "merge1 → merge2 link arg dropped after 'y' pruned"
        );
    }

    #[test]
    fn prune_dead_phis_preserves_special_blocks() {
        // Function-entry inputargs (startblock), return-value
        // (returnblock), and exception escape (exceptblock) inputargs
        // are part of the calling convention / runtime contract and
        // must NEVER be pruned even if locally unread.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        // Add an Input op at startblock that never gets read — emulates
        // an unused function parameter.
        let unused_param = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "param".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.block_mut(entry).inputargs.push(unused_param);
        graph.set_return(entry, None);

        let pre_returnblock_args = graph.block(graph.returnblock).inputargs.clone();
        let pre_exceptblock_args = graph.block(graph.exceptblock).inputargs.clone();

        prune_dead_phis(&mut graph);

        assert_eq!(
            graph.block(entry).inputargs,
            vec![unused_param],
            "startblock inputargs are calling-convention; never pruned"
        );
        assert_eq!(
            graph.block(graph.returnblock).inputargs,
            pre_returnblock_args,
            "returnblock inputargs are runtime contract; never pruned"
        );
        assert_eq!(
            graph.block(graph.exceptblock).inputargs,
            pre_exceptblock_args,
            "exceptblock inputargs are runtime contract; never pruned"
        );
    }

    #[test]
    fn prune_dead_phis_preserves_pure_raising_op() {
        // `simplify.py:435-436 canremove`'s `op is not block.raising_op`
        // clause: a pure-classified op (e.g. `int_add` modelling
        // `int_add_ovf`) parked as the LAST op of a `canraise` block
        // must NOT be DCE'd even when its result is unread.  The raise
        // side-effect is observable, so removing the op would be a
        // semantic regression.
        //
        // Shape: entry — int_add(unread result) — exitswitch=LastException;
        //   normal exit → returnblock; except exit → exceptblock.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let lhs = graph.push_op(entry, OpKind::ConstInt(1), true).unwrap();
        let rhs = graph.push_op(entry, OpKind::ConstInt(2), true).unwrap();
        let raising = graph
            .push_op(
                entry,
                OpKind::BinOp {
                    op: "int_add".into(),
                    lhs,
                    rhs,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        // Mark the entry block as canraise: exitswitch=LastException,
        // exits=[normal → returnblock, except → exceptblock].
        let returnblock = graph.returnblock;
        let exceptblock = graph.exceptblock;
        // Synthesize the (etype, evalue) pair that exceptblock expects
        // — values flow as Constants from the front-end raise site,
        // but for the test we need only the link arity to match.
        let etype = graph.push_op(entry, OpKind::ConstInt(0), true).unwrap();
        let evalue = graph.push_op(entry, OpKind::ConstInt(0), true).unwrap();
        {
            let block = graph.block_mut(entry);
            block.exitswitch = Some(ExitSwitch::LastException);
            block.exits = vec![
                Link::new_mixed(vec![LinkArg::Value(raising)], returnblock, None),
                Link::new_mixed(
                    vec![LinkArg::Value(etype), LinkArg::Value(evalue)],
                    exceptblock,
                    None,
                ),
            ];
        }

        prune_dead_phis(&mut graph);

        let still_present = graph
            .block(entry)
            .operations
            .iter()
            .any(|op| op.result == Some(raising));
        assert!(
            still_present,
            "pure raising_op (last op of canraise block) must survive DCE per \
             simplify.py:436 `op is not block.raising_op`",
        );
    }

    #[test]
    fn prune_dead_phis_skips_non_canonical_entry_blocks() {
        // Closure / generic-specialisation lowering can put function
        // parameters at a non-zero block id with no incoming link
        // (the `generated::*` pipeline does this for `eval_loop_jit`'s
        // body block).  Even if the parameter isn't read graph-side,
        // pruning must NOT remove it — the calling convention assumes
        // every inputarg is supplied.  The block is identified as
        // "entry-like" by having no link targeting it.
        let mut graph = FunctionGraph::new("test");
        // Build: startblock → returnblock(void).  Add an extra
        // orphan-entry block with one Input op + one inputarg, no
        // predecessors at all (not even from startblock).  The block
        // is unreachable but represents the "non-canonical entry"
        // shape pyre emits in some generated pipelines.
        graph.set_return(graph.startblock, None);
        let orphan_entry = graph.create_block();
        let unused_param = graph
            .push_op(
                orphan_entry,
                OpKind::Input {
                    name: "captured".into(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        graph.block_mut(orphan_entry).inputargs.push(unused_param);

        prune_dead_phis(&mut graph);

        assert_eq!(
            graph.block(orphan_entry).inputargs,
            vec![unused_param],
            "non-canonical entry block (no incoming link) must keep its inputargs"
        );
    }

    // =========================================================
    // Plan `annotator-monomorphization-tier1` phase M2.b tests —
    // CallTarget::Method transient ClassDefKey transport.
    // =========================================================

    #[test]
    fn call_target_method_default_classdef_hint_is_none() {
        // Plan M2.b invariant: outside the narrow annotate→fold window,
        // `classdef_hint` must be `None`. The public constructor is the
        // stable entry point; callers using `CallTarget::method(...)`
        // must never observe a `Some(_)` hint.
        let t = CallTarget::method("foo", Some("Bar".to_string()));
        assert_eq!(t.classdef_hint(), None);
    }

    #[test]
    fn call_target_method_with_classdef_hint_holds_and_exposes_hint() {
        use crate::annotator::description::ClassDefKey;
        let hint = ClassDefKey::from_raw(0x1234);
        let t = CallTarget::method_with_classdef_hint("foo", Some("Bar".to_string()), hint);
        assert_eq!(t.classdef_hint(), Some(hint));
        // The `name` + `receiver_root` must survive verbatim.
        match &t {
            CallTarget::Method {
                name,
                receiver_root,
                classdef_hint,
            } => {
                assert_eq!(name, "foo");
                assert_eq!(receiver_root.as_deref(), Some("Bar"));
                assert_eq!(*classdef_hint, Some(hint));
            }
            other => panic!("expected CallTarget::Method, got {other:?}"),
        }
    }

    #[test]
    fn call_target_method_classdef_hint_is_skipped_by_serde() {
        // Plan M2.b invariant: the hint must never reach serde
        // surfaces. Serializing a `Some`-hint target and deserializing
        // back must drop the hint to `None`.
        use crate::annotator::description::ClassDefKey;
        let hint = ClassDefKey::from_raw(0xDEAD);
        let t = CallTarget::method_with_classdef_hint("foo", Some("Bar".to_string()), hint);
        let json = serde_json::to_string(&t).expect("encode");
        // The serialized JSON must not contain any classdef_hint field.
        assert!(
            !json.contains("classdef_hint"),
            "serialized form leaks classdef_hint: {json}"
        );
        let round_trip: CallTarget = serde_json::from_str(&json).expect("decode");
        assert_eq!(round_trip.classdef_hint(), None);
        // But the stable surface (name, receiver_root) must survive.
        match round_trip {
            CallTarget::Method {
                name,
                receiver_root,
                classdef_hint,
            } => {
                assert_eq!(name, "foo");
                assert_eq!(receiver_root.as_deref(), Some("Bar"));
                assert_eq!(classdef_hint, None);
            }
            other => panic!("expected CallTarget::Method, got {other:?}"),
        }
    }

    /// Slice Z2.5.A — `bridge_variable` is idempotent: repeated calls
    /// with the same upstream `Variable` return the same ValueId.
    #[test]
    fn bridge_variable_idempotent_for_same_variable_identity() {
        use crate::flowspace::model::Variable;
        let mut graph = FunctionGraph::new("demo");
        let v = Variable::new();
        let vid1 = graph.bridge_variable(&v);
        let vid2 = graph.bridge_variable(&v);
        assert_eq!(vid1, vid2);
        // Cloning preserves Variable identity (id-sharing semantics in
        // `flowspace::model::Variable::clone`), so a clone bridges to
        // the same ValueId.
        let v_clone = v.clone();
        assert_eq!(graph.bridge_variable(&v_clone), vid1);
    }

    /// Slice Z2.5.A — distinct upstream `Variable`s mint distinct
    /// ValueIds via `alloc_value()`.
    #[test]
    fn bridge_variable_distinct_variables_get_distinct_valueids() {
        use crate::flowspace::model::Variable;
        let mut graph = FunctionGraph::new("demo");
        let v1 = Variable::new();
        let v2 = Variable::new();
        let vid1 = graph.bridge_variable(&v1);
        let vid2 = graph.bridge_variable(&v2);
        assert_ne!(vid1, vid2);
    }

    /// Slice Z2.5.A — fresh bridge minting advances the same allocator
    /// cursor as `alloc_value()` so ValueIds never collide between
    /// bridge-minted and IR-minted ops.
    #[test]
    fn bridge_variable_consumes_alloc_value_cursor() {
        use crate::flowspace::model::Variable;
        let mut graph = FunctionGraph::new("demo");
        let pre_alloc = graph.alloc_value();
        let v = Variable::new();
        let bridged = graph.bridge_variable(&v);
        let post_alloc = graph.alloc_value();
        assert_eq!(bridged.0, pre_alloc.0 + 1);
        assert_eq!(post_alloc.0, bridged.0 + 1);
    }
}
