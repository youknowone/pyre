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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueType {
    Int,
    /// `lltype.Unsigned` — register class is `'int'` per
    /// `getkind(Unsigned) == 'int'`, distinct from `Signed` only at
    /// the rtyper dispatch level (`rbool.py:78 uint_is_true`,
    /// `rint.py` cast family).  Downstream consumers that do not
    /// distinguish signedness (regalloc, codewriter,
    /// valuetype_to_someshell) treat `Unsigned` identically to `Int`
    /// via `Int | Unsigned` arms.
    ///
    /// Produced by the MIR front-end (`front::mir`) for Rust
    /// `u8`/`u16`/`u32`/`u64`/`usize` typed args.  Cast routing
    /// through `simple_call(__builtin__.float/bool, v_uint)`,
    /// `simple_call(rarithmetic.intmask, v_uint)`, and
    /// `simple_call(rarithmetic.r_uint, v)` lives in `front::mir`'s
    /// `Cast` lowering per
    /// `rbuiltin.py:178-189` / `rbuiltin.py:220-225` /
    /// `rarithmetic.py:600`.
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
    /// Bool literals lower in `front::mir` to the dedicated
    /// `OpKind::ConstBool(bool)` variant; the rtyper
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
    /// `lltype.Ptr(<inner>)` — pointer / GC reference family.  The
    /// optional `String` carries the Rust type-root identifier
    /// (the leading path segment with generic arguments stripped) for
    /// the pointee when the producer knows it; `None` is the
    /// un-narrowed / opaque case
    /// (mirrors `lltype.Ptr(<unspecified>)` upstream where the rtyper
    /// resolves through `getinstancerepr(rtyper, None, Gc)` to the
    /// abstract `object`-root `InstanceRepr`).
    ///
    /// `valuetype_to_someshell::Ref` intentionally projects every
    /// `Ref(_)` fallback to `SomeInstance(classdef=None)`.  Typed
    /// pointer precision must be attached before that fallback by a
    /// producer that has the real lltype object in hand, e.g. the
    /// Rust-source walker writing
    /// `SomeValue::Ptr(SomePtr::new(<ll_ptrtype>))` directly into
    /// `Variable.annotation`.  Reconstructing `SomePtr` later from this
    /// root string would lose RPython's module/global context and
    /// object-identity-based lltype semantics.
    Ref(Option<String>),
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
    /// be a deviation (treats the closure as a synchronous block).
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
        /// Resolved call path stamped at codewriter time by
        /// `stamp_classdef_hints_on_graph`. This is a synthesized
        /// [`crate::parse::CallPath`] (re-derived from the receiver's
        /// `SomeInstance.classdef` → `MethodDesc.selfclassdef` → impl
        /// type), NOT a graph-object pointer. It is pyre's surrogate for
        /// `call.py:29`'s graph-object dict key: the consumer does
        /// `function_graphs.get(&path)` directly, the `CallPath`-keyed
        /// stand-in for `getfunctionptr(graph)` (call.py:181). Transient
        /// (`#[serde(skip)]`) — never reaches codegen / serde.
        #[serde(skip, default)]
        resolved_path: Option<crate::parse::CallPath>,
    },
    FunctionPath {
        segments: Vec<String>,
    },
    /// Rust frontend adaptation for constructors that RPython's rtyper erases
    /// before jtransform. This variant must only be produced after frontend
    /// resolution proves the call is not a user-defined function.
    ///
    /// `name` is the variant/ctor leaf (`"Continue"` for
    /// `StepResult::Continue`).  `owner_path` is the qualifying segments
    /// preceding the leaf (`["StepResult"]`), empty for ctors that have
    /// no owner (`Ok`/`Err`/`Some`).  Owner path matters for downstream
    /// identity: `HostObject::new_class(qualname)` keys on the joined
    /// path, so `StepResult::Continue` and `JitAction::Continue` produce
    /// distinct ClassDescs and don't collide on the bare leaf — matching
    /// RPython `bookkeeper.py:353 getdesc(pyobj)` which dedupes by
    /// object identity, not leaf string.
    SyntheticTransparentCtor {
        name: String,
        #[serde(default)]
        owner_path: Vec<String>,
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
            resolved_path: None,
        }
    }

    pub fn resolved_path(&self) -> Option<&crate::parse::CallPath> {
        match self {
            CallTarget::Method { resolved_path, .. } => resolved_path.as_ref(),
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
        Self::SyntheticTransparentCtor {
            name: name.into(),
            owner_path: Vec::new(),
        }
    }

    /// Owner-qualified variant — preserves the full `Owner::Variant`
    /// path so downstream `HostObject` identity does not collide on
    /// the bare leaf (e.g. `StepResult::Continue` vs
    /// `JitAction::Continue`).
    pub fn synthetic_transparent_ctor_with_owner(
        owner_path: Vec<String>,
        name: impl Into<String>,
    ) -> Self {
        Self::SyntheticTransparentCtor {
            name: name.into(),
            owner_path,
        }
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
            // `(owner_path, name)` is the lookup identity per the
            // variant docs; collapsing onto the leaf alone would
            // make `Instruction::LoadFast` and `OtherEnum::LoadFast`
            // share segments and break any downstream caller that
            // uses `path_segments()` for qualified identity.
            CallTarget::SyntheticTransparentCtor { name, owner_path } => {
                let mut segs: Vec<&str> = owner_path.iter().map(String::as_str).collect();
                segs.push(name.as_str());
                Some(segs)
            }
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
            CallTarget::SyntheticTransparentCtor { name, owner_path } => {
                if owner_path.is_empty() {
                    write!(f, "<synthetic-transparent-ctor {name}>")
                } else {
                    write!(
                        f,
                        "<synthetic-transparent-ctor {}::{name}>",
                        owner_path.join("::")
                    )
                }
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
/// a symbolic direct-call target or the runtime `Variable` holding the
/// funcptr value for indirect calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallFuncPtr {
    Target(CallTarget),
    Value(crate::flowspace::model::Variable),
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDescriptor {
    pub name: String,
    pub owner_root: Option<String>,
    /// Object-identity token for the owning struct / enum-variant type,
    /// minted from the full Charon `name_path()` at the field-resolution
    /// source (`resolve_adt_field`) where the qualified path is still in
    /// hand.  The layout layer keys offsets / sizes on this token instead
    /// of the bare-leaf `owner_root` string, so two distinct type
    /// definitions sharing a leaf name resolve to distinct offsets — the
    /// analog of RPython holding the live `lltype.Struct` object.  `None`
    /// when the descriptor was built outside the typed-ADT path (synthetic
    /// tuples, tests, builder construction).
    ///
    /// Excluded from `PartialEq` / `Hash` (see the manual impls below): it
    /// is functionally determined by `owner_root` at the typed sites, and
    /// keeping it out of identity means a descriptor built with the token
    /// stays equal to / hashes with one built without it, so op dedup /
    /// CSE over `FieldDescriptor` is unaffected.
    pub owner_id: Option<majit_ir::descr::StructId>,
}

impl PartialEq for FieldDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.owner_root == other.owner_root
    }
}

impl Eq for FieldDescriptor {}

impl std::hash::Hash for FieldDescriptor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.owner_root.hash(state);
    }
}

impl FieldDescriptor {
    pub fn new(name: impl Into<String>, owner_root: Option<String>) -> Self {
        Self {
            name: name.into(),
            owner_root,
            owner_id: None,
        }
    }

    /// Builder-style setter for the owning-type identity token.
    pub fn with_owner_id(mut self, owner_id: Option<majit_ir::descr::StructId>) -> Self {
        self.owner_id = owner_id;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpKind {
    Input {
        name: String,
        ty: ValueType,
        /// Optional class identity for `Ref`-typed parameters carried
        /// from the front-end's `syn::Type` projection.  Mirrors how
        /// PyPy's annotator routes typed `&Foo` through
        /// `bookkeeper.getuniqueclassdef` (`description.py:283-305
        /// FunctionDesc.pycall`) so the rtyper's `find_attribute`
        /// (`rclass.py:556`) lands on the actual `ClassDef`.  The
        /// `front::mir` param lowering currently leaves this `None` for
        /// every parameter; typed pointer precision, when carried, comes
        /// from the leaf segment of the param's `ValueType::Ref(_)` root
        /// when that leaf matches a known struct in
        /// `program.struct_fields`.  Non-`Ref` params (Int, Float, Bool,
        /// Void) always have `None`.
        class_root: Option<String>,
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
    /// RPython `flowmodel.py:Constant(host_object)` resolved by the
    /// rtyper to a singleton instance pointer
    /// (`rtyper/rpbc.py::SingleFrozenPBCRepr`).  Stored as a thin
    /// `HostObject` handle so the assembler can stash
    /// `obj.identity_id()` in the ref-kind constant pool
    /// (`assembler.rs::emit_const_r`) and emit `ref_copy/r>r` —
    /// mirroring the `ConstInt` → `int_copy` lowering for the
    /// ref bank.  Producers: the pre-jtransform unit-variant
    /// rewrite pass (`translator/rtyper/unit_variant_fold.rs`)
    /// folds `OpKind::Call { target: SyntheticTransparentCtor,
    /// args: [] }` matching `is_synthetic_unit_variant_path`
    /// into this variant.
    ConstRef(crate::flowspace::model::HostObject),
    /// RPython `lltype.nullptr(T)` / null GC ref constant. Pyre spells
    /// the common sentinel as `PY_NULL`; it materialises in the ref bank
    /// as address 0, not as an integer register.
    ConstRefNull,
    /// Process-wide singleton pointer that must be materialised in the
    /// ref bank (e.g. PyPy `space.fromcache(DictStrategy)` singletons
    /// represented by pyre's Rust `pub static *_DICT_STRATEGY` values).
    ConstRefAddr(i64),
    /// An identity-bearing symbolic constant — pyre's model-graph
    /// carrier for RPython's `CDefinedIntSymbolic` singletons
    /// (`rpython/rlib/jit.py:360 _we_are_jitted`).  `tag` is the
    /// process-unique `flowspace::model::ConstValue::SpecTag` id the
    /// symbolic is keyed by (e.g.
    /// [`crate::translator::backendopt::constfold::WE_ARE_JITTED_TAG_ID`]);
    /// `ty` is its `concretetype` (`Bool` for pyre's
    /// `we_are_jitted() -> bool`).
    ///
    /// RPython's rtyper `specialize_call` rewrites the `direct_call`
    /// in-place to `inputconst(Signed, _we_are_jitted)`
    /// (`rpython/rlib/jit.py:403-406`), so the symbolic `Constant`
    /// flows through the single graph that both the backend and
    /// `codewriter/jtransform.py` consume.  Pyre's rtyper is a
    /// type oracle on an ephemeral graph that never rewrites the
    /// surviving model graph, so the symbolic is injected here by
    /// `front::mir` (the model-graph builder) instead.  Downstream:
    /// `flowspace_adapter` re-materialises it as `Constant(SpecTag,
    /// <ty>)` for the rtyper / `constfold::replace_we_are_jitted`
    /// (→ `false`) genc path, and `jtransform` folds it to
    /// `ConstBool(true)` keyed on the `SpecTag` identity (parity with
    /// `jtransform.py:1638 value is _we_are_jitted`).  No
    /// `ConstSymbolic` survives `jtransform`, so the backend never
    /// materialises one.
    ConstSymbolic {
        tag: u64,
        ty: ValueType,
    },
    FieldRead {
        base: crate::flowspace::model::Variable,
        field: FieldDescriptor,
        ty: ValueType,
        /// RPython `jtransform.py:867-903` may rewrite immutable /
        /// quasi-immutable reads to `getfield_*_pure`.  Carries the
        /// chosen opcode flavour through flatten/assembly so the
        /// runtime sees the `_pure` bytecode variant instead of having
        /// to rediscover purity from the descriptor later.
        pure: bool,
    },
    FieldWrite {
        base: crate::flowspace::model::Variable,
        field: FieldDescriptor,
        /// RPython `setfield_gc(p, x, descr)` where `x` is an
        /// `AbstractValue` — either a `Variable` (register operand) or a
        /// `Constant` (inline literal).  `LinkArg::Value`/`Const` mirrors
        /// that union so a small-int constant can stay inline as the
        /// stored value (`setfield_gc_i/rcd`) instead of being hoisted
        /// into a separate `ConstInt -> register` materialisation.  Until
        /// the MIR front-end stops materialising (the c-form activation),
        /// every producer mints `LinkArg::Value`.
        value: LinkArg,
        ty: ValueType,
    },
    /// RPython `malloc(STRUCT, flavor='gc')` for a fixed-size GcStruct: the
    /// heap allocation of a boxed object (`pyre_object::lltype::malloc_typed`).
    /// Lowered to the `new_with_vtable` jitcode op (executor
    /// `OpCode::NewWithVtable`). `owner` is the struct leaf (e.g.
    /// `"W_FloatObject"`); the assembler resolves the size descriptor from it
    /// via `bh_size_spec_from_callcontrol`, whose `path_hash(owner)` keys the
    /// runtime `gc_cache._cache_size` Arc carrying the struct size + gc
    /// type-id.
    ///
    /// `vtable` is the type-pointer (the `&FLOAT_TYPE` / `&INT_TYPE` /
    /// `&COMPLEX_TYPE` static address) the runtime stamps into the fresh
    /// object's `ob_type` (and, via `get_instantiate`, its `w_class`).  The GC
    /// allocator reads it from the size descriptor's `vtable` field — NOT from
    /// `type_id`, which resolves the struct *size* only (`gc_cache._cache_size`
    /// carries no type pointer).  `fuse_boxing_alloc` captures it from the
    /// boxed constructor's dropped `ob_header.ob_type` store; a `0` vtable is
    /// an unresolved/non-boxing placeholder the assembler rejects.  The result
    /// register is always a fresh `Ref` ('r').
    NewWithVtable {
        owner: String,
        vtable: i64,
    },
    ArrayRead {
        base: crate::flowspace::model::Variable,
        index: crate::flowspace::model::Variable,
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
        /// RPython: the foldable/immutable element load
        /// `getarrayitem_gc_{i,r,f}_pure`.  `ll_getitem_foldable_nonneg`
        /// (rlist.py:721-724, `oopspec = 'list.getitem_foldable(l,
        /// index)'`) is selected by `rtype_getitem` when `not
        /// listdef.listitem.mutated` (rlist.py:256-258), and the
        /// `FixedSizeListRepr` iterator next is `ll_listnext_foldable`
        /// when `not r_list.listitem.mutated`
        /// (lltypesystem/rlist.py:462-466).  `true` only for such a
        /// foldable/unmutated selection or an immutable container —
        /// NEVER for a mutable list element.
        pure: bool,
    },
    /// `arraylen_gc(array, arraydescr)` — read a GcArray's length
    /// header (`len(l.items)`, rlist.py:251).  Carries only the array
    /// base; the result is always the length `Int`.
    ArrayLen {
        base: crate::flowspace::model::Variable,
        /// ARRAY identity for `cpu.arraydescrof(ARRAY)`.
        array_type_id: Option<String>,
        /// `ARRAY_INSIDE._hints.get('nolength', False)` (descr.py:359).
        /// `false` (length-prefixed) lays the length word at offset 0.
        nolength: bool,
    },
    ArrayWrite {
        base: crate::flowspace::model::Variable,
        index: crate::flowspace::model::Variable,
        /// RPython `setarrayitem_gc(p, i, x, descr)` where `x` is an
        /// `AbstractValue` — `op.args[2]` may be a `Variable` or a
        /// `Constant` (jtransform.py:803).  `LinkArg::Value`/`Const`
        /// mirrors that union so a small-int constant value can stay
        /// inline (`setarrayitem_gc_i/ricd`) instead of being hoisted
        /// into a `ConstInt -> register` materialisation, matching the
        /// FieldWrite c-form activation.
        value: LinkArg,
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
        base: crate::flowspace::model::Variable,
        index: crate::flowspace::model::Variable,
        field: FieldDescriptor,
        item_ty: ValueType,
        array_type_id: Option<String>,
    },
    /// RPython: setinteriorfield_gc — write a field of an array-of-structs element.
    /// effectinfo.py:349-350: generates "interiorfield" effect.
    /// effectinfo.py:327-340: also implicitly generates "array" effect.
    InteriorFieldWrite {
        base: crate::flowspace::model::Variable,
        index: crate::flowspace::model::Variable,
        field: FieldDescriptor,
        value: crate::flowspace::model::Variable,
        item_ty: ValueType,
        array_type_id: Option<String>,
    },
    Call {
        target: CallTarget,
        args: Vec<crate::flowspace::model::Variable>,
        result_ty: ValueType,
    },
    GuardTrue {
        cond: crate::flowspace::model::Variable,
    },
    GuardFalse {
        cond: crate::flowspace::model::Variable,
    },

    // ── JIT-specific ops (generated by jtransform pass) ──────────
    /// Guard that a value equals a compile-time constant.
    ///
    /// RPython upstream emits three opnames in the guard_value family
    /// via `codewriter/jtransform.py:608-614 rewrite_op_hint`:
    /// `int_guard_value` / `ref_guard_value` / `float_guard_value`,
    /// each a 1-input/0-output pointer-or-value compare per the arg's
    /// `getkind()`.
    ///
    /// `str_guard_value` (`jit.py:631` for `promote_string`, `:647`
    /// for `promote_unicode`) is NOT modeled as a distinct op: it is the
    /// value-equality promotion of an `rstr.STR` / `rstr.UNICODE` low-level
    /// string, comparing the inline char array
    /// (`support.py:526-538 _ll_2_str_eq_nonnull` indexes `s.chars[i]`).
    /// pyre interpreter strings are `W_UnicodeObject` GC refs, never
    /// `Ptr(rstr.STR)`, so there is no char array to value-compare; the
    /// `PromoteString` / `PromoteUnicode` rewrite arms lower their ref
    /// operand through the ref-kind member of this family — `r_guard_value`
    /// (`kind_char == 'r'`), an identity guard on the string pointer.
    GuardValue {
        value: crate::flowspace::model::Variable,
        /// `'i'` int, `'r'` ref, `'f'` float — matching the
        /// `<kind>_guard_value` family naming at
        /// `codewriter/jtransform.py:611`.
        kind_char: char,
    },
    /// Project a callee function pointer out of a `dyn Trait` receiver's
    /// vtable for the named method slot.  Result is integer-typed so it
    /// can be fed to `int_guard_value` (RPython `jtransform.py:546`).
    ///
    /// TODO: pyre adaptation of `rclass.py:371-377 getclsfield()`. RPython
    /// emits a `cast_pointer + getfield(vtable_struct, method_name)` chain
    /// because `ClassRepr` models the vtable as an explicit `Struct`. Rust
    /// `dyn Trait` vtable layout is compiler-internal (unstable ABI), so
    /// pyre cannot model the vtable as an IR struct — this single op stands
    /// in for the chain and must be emitted by the rtyper-equivalent layer
    /// (`translator/rtyper/rclass.rs`), never by `jtransform`.
    VtableMethodPtr {
        receiver: crate::flowspace::model::Variable,
        trait_root: String,
        method_name: String,
    },
    /// Indirect call — `funcptr` is the `Variable` carrying the callable
    /// produced by the rtyper layer (e.g. from `VtableMethodPtr` for
    /// `dyn Trait` dispatch). `args` are the full call arguments,
    /// including the receiver. `graphs` mirrors the trailing `c_graphs`
    /// constant from `rpbc.py:216`: `Some(full_family)` when known,
    /// `None` otherwise.
    ///
    /// RPython: `rpython/rtyper/rpbc.py:216-217`
    /// ```python
    /// vlist.append(hop.inputconst(Void, row_of_graphs.values()))
    /// v = hop.genop('indirect_call', vlist, resulttype=rresult)
    /// ```
    /// Lowered downstream by `jtransform.py:410-412 rewrite_op_indirect_call`.
    IndirectCall {
        funcptr: crate::flowspace::model::Variable,
        args: Vec<crate::flowspace::model::Variable>,
        graphs: Option<Vec<crate::parse::CallPath>>,
        result_ty: ValueType,
    },
    /// Virtualizable field read → reads from boxes, no heap op.
    /// RPython: `getfield_vable_i/r/f`
    VableFieldRead {
        base: crate::flowspace::model::Variable,
        field_index: usize,
        ty: ValueType,
    },
    /// Virtualizable field write → writes to boxes, no heap op.
    /// RPython: `setfield_vable_i/r/f`
    ///
    /// `value` is a [`LinkArg`] (register or inline constant) like
    /// [`OpKind::FieldWrite`]: `rewrite_op_setfield` passes the setfield
    /// `v_value` straight to `setfield_vable_%s`
    /// (`jtransform.py:921-927`), which may be a `Constant`
    /// (`flatten.py:360-371`).  `setfield_vable_i` is not in `USE_C_FORM`
    /// (`assembler.py:312-345`), so a constant value always takes the
    /// pool `i` slot (`setfield_vable_i/rid`), never the short `c` byte.
    VableFieldWrite {
        base: crate::flowspace::model::Variable,
        field_index: usize,
        value: LinkArg,
        ty: ValueType,
    },
    /// Virtualizable array read → reads from boxes.
    /// RPython: `getarrayitem_vable_i/r/f`
    VableArrayRead {
        base: crate::flowspace::model::Variable,
        array_index: usize,
        elem_index: crate::flowspace::model::Variable,
        item_ty: ValueType,
        /// RPython: arraydescr.itemsize from VirtualizableInfo.array_descrs.
        array_itemsize: usize,
        /// RPython: arraydescr.is_item_signed() from VirtualizableInfo.array_descrs.
        array_is_signed: bool,
    },
    /// Virtualizable array write → writes to boxes.
    /// RPython: `setarrayitem_vable_i/r/f`
    VableArrayWrite {
        base: crate::flowspace::model::Variable,
        array_index: usize,
        elem_index: crate::flowspace::model::Variable,
        value: crate::flowspace::model::Variable,
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
        lhs: crate::flowspace::model::Variable,
        rhs: crate::flowspace::model::Variable,
        result_ty: ValueType,
    },
    /// Unary operation.
    /// RPython: `int_neg`, `bool_not`, etc.
    UnaryOp {
        op: String,
        operand: crate::flowspace::model::Variable,
        result_ty: ValueType,
    },

    /// Force virtualizable: flush boxes to heap.
    /// RPython: `hint_force_virtualizable(vable)`
    VableForce {
        base: crate::flowspace::model::Variable,
    },

    /// JIT hint operation — `hint(x, **kwds)`.
    ///
    /// RPython models a single `hint` operator
    /// (`rpython/flowspace/operation.py:521 add_operator('hint', None,
    /// dispatch=1)`) whose kwarg dict selects the behaviour; `rlib/jit.py:101
    /// promote(x)` and `#[elidable_promote]`'s per-arg wrapper both lower to
    /// it.  Pyre carries the kwarg key as the structured [`kind`] field
    /// instead of a `Void`-constant dict (which `OpKind::Call` cannot hold),
    /// so the typing oracle (`flowspace_adapter`) and the JIT codewriter
    /// (`jtransform::rewrite_op_hint`) dispatch on the `kind` rather than a
    /// synthesised marker name.  Outside the JIT the op is an identity on
    /// `value` (the flowspace oracle lowers it to `same_as`); the JIT
    /// codewriter rewrites it to the `<kind>_guard_value` family
    /// (`codewriter/jtransform.py:608-614`).
    Hint {
        value: crate::flowspace::model::Variable,
        kind: crate::hints::HintKind,
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
    /// symbolic target; indirect calls carry the runtime `Variable`
    /// produced by rtype.
    CallElidable {
        funcptr: CallFuncPtr,
        descriptor: crate::call::CallDescriptor,
        args_i: Vec<crate::flowspace::model::Variable>,
        args_r: Vec<crate::flowspace::model::Variable>,
        args_f: Vec<crate::flowspace::model::Variable>,
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
        args_i: Vec<crate::flowspace::model::Variable>,
        args_r: Vec<crate::flowspace::model::Variable>,
        args_f: Vec<crate::flowspace::model::Variable>,
        result_kind: char,
        indirect_targets: Option<IndirectCallTargets>,
    },
    /// May-force call — can trigger GC or force virtualizables.
    /// RPython: `call_may_force_{kinds}_{reskind}(funcptr, calldescr, [i], [r], [f])`.
    /// See `CallElidable` for `funcptr` semantics.
    CallMayForce {
        funcptr: CallFuncPtr,
        descriptor: crate::call::CallDescriptor,
        args_i: Vec<crate::flowspace::model::Variable>,
        args_r: Vec<crate::flowspace::model::Variable>,
        args_f: Vec<crate::flowspace::model::Variable>,
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
        args_i: Vec<crate::flowspace::model::Variable>,
        /// Reference arguments (RPython: ListOfKind('ref', ...))
        args_r: Vec<crate::flowspace::model::Variable>,
        /// Float arguments (RPython: ListOfKind('float', ...))
        args_f: Vec<crate::flowspace::model::Variable>,
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
        greens_i: Vec<crate::flowspace::model::Variable>,
        greens_r: Vec<crate::flowspace::model::Variable>,
        greens_f: Vec<crate::flowspace::model::Variable>,
        /// Red args (loop-variant) split by kind
        reds_i: Vec<crate::flowspace::model::Variable>,
        reds_r: Vec<crate::flowspace::model::Variable>,
        reds_f: Vec<crate::flowspace::model::Variable>,
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
        args: Vec<crate::flowspace::model::Variable>,
    },
    /// jtransform.py:1733 — `{kind}_assert_green(value)`.
    /// Asserts the value is compile-time constant during tracing.
    AssertGreen {
        value: crate::flowspace::model::Variable,
        kind_char: char,
    },
    /// jtransform.py:1736 — `current_trace_length()`.
    /// Returns the current length of the trace being compiled.
    CurrentTraceLength,
    /// jtransform.py:1738 — `{kind}_isconstant(value)`.
    /// Returns whether the value is currently known to be constant.
    IsConstant {
        value: crate::flowspace::model::Variable,
        kind_char: char,
    },
    /// jtransform.py:1741 — `{kind}_isvirtual(value)`.
    /// Returns whether the value is currently virtualized.
    IsVirtual {
        value: crate::flowspace::model::Variable,
        kind_char: char,
    },
    /// RPython `flowspace` `isinstance(obj, cls)` (annotator/unaryop.py +
    /// rtyper.rs:2035 dispatch — `Repr.rtype_isinstance`). The
    /// front-end emits this op at `match` cascade sites where the
    /// variant pattern carries a payload (`TupleStruct`) and a
    /// ptr_eq-against-singleton check would be insufficient.
    /// `class_carrier` is typically a `ConstRef`-wrapped vtable
    /// Constant resolved by the rtyper to a CLASSTYPE pointer; the
    /// rtyper then dispatches to
    /// [`InstanceRepr::rtype_isinstance`](crate::translator::rtyper::rclass::InstanceRepr::rtype_isinstance)
    /// which mints either `ll_isinstance_const_{,nonnull}_<flavor>_<class_identity>`
    /// (Constant `class_carrier`) or the generic `ll_isinstance`
    /// (Variable `class_carrier`).
    IsInstance {
        obj: crate::flowspace::model::Variable,
        class_carrier: crate::flowspace::model::Variable,
        result_ty: ValueType,
    },

    // ── Conditional call ops (jtransform.py:1665-1688) ──────
    //
    // RPython: `jit_conditional_call` / `jit_conditional_call_value` llops
    // are rewritten to `conditional_call_{kinds}_{reskind}`.
    /// jtransform.py:1685 — `conditional_call_{ir}_{v}`.
    /// If condition is true, call the function. Always produces void.
    /// RPython: `COND_CALL(condition, funcptr, calldescr, args...)`
    ConditionalCall {
        condition: crate::flowspace::model::Variable,
        funcptr: CallTarget,
        descriptor: crate::call::CallDescriptor,
        args_i: Vec<crate::flowspace::model::Variable>,
        args_r: Vec<crate::flowspace::model::Variable>,
        args_f: Vec<crate::flowspace::model::Variable>,
    },
    /// jtransform.py:1687 — `conditional_call_value_{ir}_{reskind}`.
    /// If value is falsy (0/NULL/None), call the function and return its result.
    /// RPython: `COND_CALL_VALUE(value, funcptr, calldescr, args...)`
    ConditionalCallValue {
        value: crate::flowspace::model::Variable,
        funcptr: CallTarget,
        descriptor: crate::call::CallDescriptor,
        args_i: Vec<crate::flowspace::model::Variable>,
        args_r: Vec<crate::flowspace::model::Variable>,
        args_f: Vec<crate::flowspace::model::Variable>,
        result_kind: char,
    },

    /// jtransform.py:292-313 — `record_known_result_{i|r}_ir_v`.
    /// Produced by `rewrite_op_jit_record_known_result`; pairs an elidable call
    /// with its known result for constant folding by OptPure.
    /// RPython layout: `record_known_result_{reskind}(result, funcptr, calldescr, [i], [r])`
    RecordKnownResult {
        /// The known result value (arg 0 of the jit_record_known_result llop).
        result_value: crate::flowspace::model::Variable,
        funcptr: CallTarget,
        descriptor: crate::call::CallDescriptor,
        args_i: Vec<crate::flowspace::model::Variable>,
        args_r: Vec<crate::flowspace::model::Variable>,
        args_f: Vec<crate::flowspace::model::Variable>,
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
    /// TODO: RPython derives `mutate_field` via
    /// `quasiimmut.get_mutate_field_name(name)` which expects the lltype
    /// `inst_` prefix (`quasiimmut.py:11-15`).  Rust structs have no such
    /// prefix, so we use the literal `mutate_<fieldname>` convention.
    RecordQuasiImmutField {
        base: crate::flowspace::model::Variable,
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
        greens_i: Vec<crate::flowspace::model::Variable>,
        greens_r: Vec<crate::flowspace::model::Variable>,
        greens_f: Vec<crate::flowspace::model::Variable>,
        reds_i: Vec<crate::flowspace::model::Variable>,
        reds_r: Vec<crate::flowspace::model::Variable>,
        reds_f: Vec<crate::flowspace::model::Variable>,
    },

    /// JitDriver loop-header marker — RPython `loop_header` opname.
    /// Emitted by `handle_jit_marker__loop_header` (jtransform.py:1714-1718)
    /// with the jitdriver's index as its single Constant arg.
    /// `can_enter_jit` markers alias to this (jtransform.py:1723).
    LoopHeader {
        jitdriver_index: usize,
    },

    /// pyre-only marker emitted by `front::opcode_wrapper` when an
    /// opcode-dispatch arm cannot be lowered to a canonical opname.
    /// Reaching the op at runtime means tracing or blackhole resume
    /// crossed an untranslatable graph slice; downstream handlers
    /// advance past it (see `blackhole.rs::handler_abort_marker_pyre`).
    /// Distinct from RPython's `SwitchToBlackhole` exception path —
    /// RPython aborts before lowering so no equivalent opname exists.
    /// Kept under `kind: UnknownKind` because the same diagnostic enum
    /// is reused by `front::mir`'s `LowerError::Unsupported`.
    Abort {
        kind: UnknownKind,
    },

    /// RPython `BUILD_TUPLE` (`flowspace/flowcontext.py:1163`) /
    /// `newtuple` operation (`operation.py:542-548`).  Constructs a
    /// new tuple from N element values; the result is a fresh tuple
    /// object distinct from any individual element.
    NewTuple {
        args: Vec<crate::flowspace::model::Variable>,
    },

    /// RPython `BUILD_LIST` (`flowspace/flowcontext.py`) / `newlist`
    /// operation (`operation.py`).  Constructs a fresh resizable list
    /// from N element values; the result is a new list object distinct
    /// from any individual element.
    NewList {
        args: Vec<crate::flowspace::model::Variable>,
    },

    /// A pre-lowered, register-shaped blackhole opcode emitted by the
    /// opname-dispatch convergence spine (`codewriter::jtransform_opname`).
    ///
    /// The rtyper lowers certain helper graphs (the `ll_str*` family) to
    /// upstream-shaped low-level `SpaceOperation`s whose opnames map 1:1 onto
    /// a register-only blackhole insn with NO descriptor operand — e.g.
    /// `strlen` (`r>i`), `strgetitem` (`ri>i`), `strsetitem` (`rii`),
    /// `newstr` (`i>r`).  Rather than mint a distinct rich `OpKind` per such
    /// opname (each forcing an arm in every exhaustive `OpKind` match), the
    /// transducer carries the blackhole `opname` and its operand `Variable`s
    /// directly here.  The assembler's default `encode_op` arm emits
    /// `{opname}/{argcodes}`, with operand kinds inferred from each operand's
    /// register bank and the result kind (if any) appended from `op.result`
    /// — matching the `bhimpl_<opname>` argcode shape the runtime handler is
    /// keyed on (`blackhole.rs`).
    ///
    /// Only for opnames whose lowering is a single register-shaped insn with
    /// no descriptor operand; effect classification remains opname-driven
    /// (e.g. `newstr`/`newunicode` are `MemoryErrorOnly` in
    /// `codewriter::call::op_can_raise`).  Descriptor-bearing ops
    /// (getfield/getarrayitem/…) keep their dedicated rich `OpKind`s.
    LoweredBlackholeOp {
        opname: String,
        args: Vec<crate::flowspace::model::Variable>,
    },

    /// A single-segment path resolving to a crate-local `static`
    /// declaration (typically a SHOUTY_CASE constant like
    /// `GC_WEAKREF_BOX_TYPE`, `INT_TYPE`, `MODULE_DICT_TYPE`).
    ///
    /// RPython parity: `flowspace/flowcontext.py:LOAD_GLOBAL` lifts a
    /// module-scope name lookup to a `Constant(value)` whose payload
    /// is the resolved object (`flowspace/flowcontext.py:1098`); the
    /// annotator binds the result via `unionof(s_Constant)` without
    /// emitting a SpaceOperation (the bound `Variable` *is* the
    /// graph-level definition).  Pyre's adapter emits a placeholder
    /// `same_as` SpaceOperation whose first arg is the static's
    /// declared `Hlvalue::Constant` so checkgraph's defining-var
    /// set includes the producer (this differs from PyPy where
    /// `LOAD_GLOBAL` returns a Constant Hlvalue directly and no
    /// SpaceOperation is needed — pyre's frontend always emits an
    /// op so cross-block reads have a defined producer).
    ///
    /// PRE-EXISTING-ADAPTATION: RPython has no concept of a
    /// "static at a crate path" — it has module-global Python names
    /// looked up by `LOAD_GLOBAL`.  The `segments` field carries the
    /// fully-qualified path (`["module_path", "STATIC_NAME"]`)
    /// matching the upstream lookup key shape.
    ///
    /// Slice C — `value` carries the literal-evaluated initializer
    /// when `extract_static_decls` could fold the RHS to a
    /// `ConstValue` (bool/int/float/string literals, including the
    /// `-LIT` unary-neg shape and the `thread_local!` `const { LIT }`
    /// wrapper).  `Some(v)` reaches the flowspace adapter as the
    /// concrete `Constant(v)` operand of the synthetic `same_as` op,
    /// matching PyPy `LOAD_GLOBAL` (`flowcontext.py:856`) pushing the
    /// resolved object.  `None` is rejected before JitCode assembly:
    /// RPython has no blackhole `same_as/*` opcode, so unresolved
    /// statics must be folded to constants or lowered through a real
    /// host-evaluator path first.
    LoadStatic {
        segments: Vec<String>,
        ty: ValueType,
        value: Option<crate::flowspace::model::ConstValue>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpaceOperation {
    /// RPython `flowspace/model.py:439 self.result = result      #
    /// either Variable or Constant instance`.  In the
    /// rtyper-orthodox path the canonical writer is
    /// `flowspace/operation.py:75 self.result = Variable()`,
    /// matching the `Option<Variable>` carrier — `None` reserved for
    /// the `direct_call`-shape test fixtures whose result slot is
    /// unbound (`rpython/translator/backendopt/test/test_graphanalyze.py:76`).
    pub result: Option<crate::flowspace::model::Variable>,
    pub kind: OpKind,
}

/// RPython `Block.exitswitch`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExitSwitch {
    Value(crate::flowspace::model::Variable),
    LastException,
    /// `jtransform.py:196-234 optimize_goto_if_not` fuses a comparison
    /// op into the exitswitch: `block.exitswitch = (opname,) +
    /// tuple(op.args) + ('-live-before',)`.  Carries the RPython
    /// comparison opname (e.g. `int_lt`) and the operands; the trailing
    /// `-live-before` marker is implicit (always present for a fused
    /// guard) and re-applied at emit time (`flatten.py:248-253`).
    Fused {
        opname: String,
        args: Vec<crate::flowspace::model::Variable>,
    },
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
    /// `Variable`-typed constructor — direct counterpart to RPython
    /// `Link(args=[Hlvalue], target=Block, exitcase=...)` where each
    /// `Hlvalue` is the upstream `Variable` instance pulled from an
    /// `op.result` / `Block.inputargs` slot.  Mirrors the upstream
    /// `flowspace/model.py:114-116` arity assert via `graph` lookup
    /// for `target.inputargs`.
    pub fn from_variables(
        graph: &FunctionGraph,
        args: Vec<crate::flowspace::model::Variable>,
        target: BlockId,
        exitcase: Option<ExitCase>,
    ) -> Self {
        assert_eq!(
            args.len(),
            graph.block(target).inputargs.len(),
            "output args mismatch"
        );
        Self::new_mixed(
            args.into_iter().map(LinkArg::Value).collect(),
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

/// RPython `Link.args` items are Variables or Constants —
/// `Link.args: List[Hlvalue]` (`flowspace/model.py:140`) where
/// `Hlvalue = Variable | Constant`.  Both arms carry the
/// upstream-orthodox flowspace handles directly: `Value` wraps a
/// [`crate::flowspace::model::Variable`] (concretetype on the inline
/// cell) and `Const` wraps a [`crate::flowspace::model::Constant`]
/// (concretetype on the struct field) so type-sensitive renaming /
/// Void filtering at link sites can read kinds off either arm without
/// projecting through a side table.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LinkArg {
    Value(crate::flowspace::model::Variable),
    Const(crate::flowspace::model::Constant),
}

impl LinkArg {
    /// Read the backing [`crate::flowspace::model::Variable`] for a
    /// `LinkArg::Value`; `None` for constants.  Direct over the
    /// upstream-orthodox storage — no graph projection needed.
    ///
    /// RPython parity: `Link.args` upstream is `List[Hlvalue]` where
    /// each `Hlvalue::Variable` carries the operand identity inline
    /// (`flowspace/model.py:140`).
    pub fn as_variable(&self) -> Option<&crate::flowspace::model::Variable> {
        match self {
            Self::Value(var) => Some(var),
            Self::Const(_) => None,
        }
    }

    /// Remap the backing `Variable` of a `LinkArg::Value` through
    /// `remap`, leaving a `LinkArg::Const` literal untouched.  Mirrors
    /// the `remap_link_arg` closure in
    /// [`remap_control_flow_metadata_var`] and RPython
    /// `Hlvalue.replace(mapping)` where a `Constant` returns itself.
    pub fn map_value(
        &self,
        remap: impl FnOnce(&crate::flowspace::model::Variable) -> crate::flowspace::model::Variable,
    ) -> Self {
        match self {
            Self::Value(var) => Self::Value(remap(var)),
            Self::Const(value) => Self::Const(value.clone()),
        }
    }
}

impl From<ConstValue> for LinkArg {
    /// Wrap a raw [`ConstValue`] in a [`crate::flowspace::model::Constant`]
    /// with no concretetype attached.  Mirrors RPython
    /// `Constant(value)` (`flowspace/model.py:354 __init__`) where
    /// `concretetype` defaults to `None` until the rtyper sets it.
    fn from(value: ConstValue) -> Self {
        Self::Const(crate::flowspace::model::Constant::new(value))
    }
}

impl From<crate::flowspace::model::Constant> for LinkArg {
    fn from(c: crate::flowspace::model::Constant) -> Self {
        Self::Const(c)
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
    ///
    /// RPython parity: `Block.inputargs: List[Variable]`
    /// (`flowspace/model.py:21-25 Block([Variable("etype"),
    /// Variable("evalue")])`) — each predecessor Link carries
    /// values that map 1:1 to these inputargs.  Pyre stores the
    /// upstream-orthodox `Variable` directly; consumers match on the
    /// `Variable` object by identity.
    pub inputargs: Vec<crate::flowspace::model::Variable>,
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
    /// fact, what `(name → Variable)` mapping was visible to a
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
/// graphs are lowered from Charon ULLBC rather than Python bytecode, so
/// the `stack` / `last_exception` / `blocklist` / `next_offset`
/// projections are vestigially empty.
///
/// All five fields are present in the struct now so
/// shape parity is locked in at the model layer; downstream merging
/// passes already thread them via `union`.  The flowcontext walker
/// (`flowspace::flowcontext::FlowContext`) and `flowspace::framestate::
/// FrameState` are the upstream-orthodox surfaces this struct is
/// converging toward — eventually flowspace graphs are produced directly
/// and this slot-indexed `FrameState` retires.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FrameState {
    /// Slot `i` ↔ graph-wide first-bind name at index `i`; `None` if
    /// unbound at this snapshot point.  RPython parity:
    /// `framestate.py:19 self.locals_w` — list of `Variable | Constant
    /// | None` indexed by `co_varnames` slot.  Carries upstream
    /// `Variable` identity directly; the parallel `locals_w`
    /// `Vec<Option<Hlvalue>>` carrier (next field) still distinguishes
    /// `Hlvalue::Constant` cells which this Variable-only carrier
    /// collapses to `None`.
    pub entries: Vec<Option<crate::flowspace::model::Variable>>,
    /// Parallel `Hlvalue` carrier matching upstream
    /// `framestate.py:19 self.locals_w` shape — list of
    /// `Variable | Constant | None` indexed by `co_varnames` slot.
    /// Populated by `getstate` / `union` alongside `entries`; the
    /// long-term plan is to promote this to the single source of truth
    /// and retire the per-entry `Variable` carrier in `entries`.  Production read
    /// sites still consume `entries`; future substeps swap them over
    /// slot-by-slot before flipping the construction direction
    /// (fixtures populate `locals_w` directly, `entries` becomes
    /// the derived projection).
    ///
    /// Fixtures that build a `FrameState` by hand keep this empty;
    /// only `FrameState::union` populates it today.  Read sites that
    /// rely on it must therefore route through a unioned state, not
    /// a hand-built fixture.
    pub locals_w: Vec<Option<crate::flowspace::model::Hlvalue>>,
    /// `framestate.py:21 self.stack` — value-stack content at the
    /// snapshot point.  Empty for AST-frontend snapshots until the
    /// flowcontext-style walker introduces stack push/pop on Expr
    /// nodes; the `union` invariant requires both predecessors agree
    /// on stack content (upstream `framestate.py:79 _union(self.stack,
    /// other.stack)` zips positionally — equal-length always for any
    /// program reachable to a join).
    pub stack: Vec<crate::flowspace::framestate::StackElem>,
    /// `framestate.py:22 self.last_exception` — pending FSException at
    /// the snapshot point.  None for AST-frontend snapshots until
    /// the flowcontext-style walker introduces exception handling.
    pub last_exception: Option<crate::flowspace::model::FSException>,
    /// `framestate.py:23 self.blocklist` — block-stack snapshot
    /// (`SETUP_*` / `POP_BLOCK` depth at the snapshot point).  Empty
    /// for AST-frontend snapshots until the flowcontext-style walker
    /// introduces frame-block management.  `framestate.py:58 matches` asserts
    /// blocklist equality across merge candidates as a precondition.
    pub blocklist: Vec<crate::flowspace::flowcontext::FrameBlock>,
    /// `framestate.py:24 self.next_offset` — bytecode offset resumed
    /// at after the snapshot.  `0` for AST-frontend snapshots until
    /// the flowcontext-style walker uses an AST-node index (the equivalent of a
    /// virtual-bytecode tape position).  `framestate.py:59 matches`
    /// asserts next_offset equality across merge candidates as a
    /// precondition.
    pub next_offset: i64,
}

impl FrameState {
    /// Read the backing `Variable` bound at `entries[slot]`.  `Some`
    /// when the slot is bound; `None` for unbound (None-killed) slots
    /// or out-of-range indices.  `entries` holds `Variable`s directly,
    /// so this is a plain positional read.
    pub fn entry_var(
        &self,
        slot: usize,
        _graph: &FunctionGraph,
    ) -> Option<crate::flowspace::model::Variable> {
        self.entries.get(slot).cloned().flatten()
    }

    /// Authoritative locals view — upstream `framestate.py:19 self.locals_w`
    /// IS the locals source-of-truth.  Pyre's `union` / `getstate`
    /// constructors populate `self.locals_w` in lockstep with
    /// `self.entries` so production callers consult the `Hlvalue` carrier
    /// directly via this view.  Hand-built fixtures that pre-date the
    /// Z4.A.6 carrier swap may leave `self.locals_w` empty; in that case
    /// derive the view from `self.entries` here so a unioned and a
    /// fixture-built FrameState produce the same `mergeable` projection
    /// without each read site re-implementing the fallback.  The
    /// derivation mirrors what `FrameState::union` and
    /// `GraphBuildContext::getstate` already do, so the result is
    /// bit-identical to the carrier they would have produced.
    pub(crate) fn locals_w_view<'a>(
        &'a self,
        graph: &FunctionGraph,
    ) -> std::borrow::Cow<'a, [Option<crate::flowspace::model::Hlvalue>]> {
        if self.locals_w.len() == self.entries.len() {
            std::borrow::Cow::Borrowed(&self.locals_w)
        } else {
            std::borrow::Cow::Owned(
                (0..self.entries.len())
                    .map(|i| {
                        self.entry_var(i, graph)
                            .map(crate::flowspace::model::Hlvalue::Variable)
                    })
                    .collect(),
            )
        }
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
    ///   - `(Some(s), Some(o))` with matching Variable identity →
    ///     carry through that Variable (`framestate.py:108 if w1 == w2:
    ///     return w1`).
    ///   - `(Some(s), Some(o))` with disagreeing Variables →
    ///     `graph.alloc_value_var()` for a fresh slot at this
    ///     position (`framestate.py:113-114 return Variable()`
    ///     analogue).
    ///
    /// Type unification is NOT performed here — upstream's per-slot
    /// `union(w1, w2)` (`framestate.py:105-128`) compares Hlvalue
    /// identity only, with type-side reconciliation deferred to the
    /// annotator (`annrpython.py`).  Pyre follows the same convention:
    /// callers query types via `FunctionGraph::concretetype_of(&var)`
    /// at the point of use; the prior per-slot `value_type` field on
    /// `FrameStateEntry` was a deviation that has been retired.
    ///
    /// Returns `Some(merged_state)` when the union succeeds; `None`
    /// when any per-projection union raises `UnionError`
    /// (`framestate.py:78 try: ... except UnionError: return None`).
    /// The `None` return propagates upstream's "merge candidates
    /// disagree, fall back to a fresh SpamBlock" path at
    /// `flowcontext.py:431-436 mergeblock`.  Agreement slots in the
    /// locals projection carry the predecessor's Variable, disagreement
    /// slots get a freshly-allocated Variable (the upstream `Variable()`
    /// analogue).  Callers detect "this slot is a fresh phi" by
    /// comparing the merged Variable identity against the predecessor's
    /// for the same slot.  The install is then driven via
    /// `lazy_install_local_at_current_block(.., Some(merged_var))` so
    /// the Input op carries the same Variable the merged state already
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
        // frontend keeps both projections at trivial defaults
        // today (empty Vec, next_offset 0); even when they diverge,
        // upstream still produces a merged FrameState that just
        // carries `self.{blocklist,next_offset}` and lets `matches`
        // catch the mismatch downstream.
        //
        // Body order (`framestate.py:79-87`): locals → stack →
        // exception, all inside the try/except envelope.  Pyre
        // **reorders** to (stack → exception → locals): the locals
        // fold is total (Variable domain has no UnionError analogue)
        // but invokes `graph.alloc_value_var()` which advances a
        // global allocator counter.  Stack and exception unions CAN
        // return `None`/`Err` once the Z4 walker activates real
        // cells; doing them first keeps `union` atomic — no
        // `alloc_value_var` writes unless the whole merge succeeds.
        // Upstream's `Variable()`
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
        // Run LAST so the locals projection is computed only after the
        // failure-prone stack / exception projections above have
        // succeeded (each `.ok()?` aborts the whole merge on a per-cell
        // `UnionError`).
        //
        // Direct line-by-line port: walk `self.locals_w` and
        // `other.locals_w` positionally and dispatch each cell pair
        // through `flowspace::framestate::union` (Hlvalue-domain
        // per-cell union — `framestate.py:105-128`).  Upstream
        // `_union(seq1, seq2)` zips equal-length sequences; pyre's
        // frontend can grow the locals projection across the
        // sub-block boundary when a name is first bound on one side,
        // so we extend to `max(len, len)` and treat the trailing
        // unpadded slot as `None` (per-cell `union` returns
        // `Ok(None)` when either side is `None`, matching upstream
        // `framestate.py:110-111` undefined-local kill).
        let self_view = self.locals_w_view(graph);
        let other_view = other.locals_w_view(graph);
        let len = std::cmp::max(self_view.len(), other_view.len());
        // Per `framestate.py:78-89`, a `UnionError` raised by ANY
        // per-cell `union(w1, w2)` is caught at the FrameState level
        // and the whole `FrameState.union` returns `None`.  Collect
        // into `Result<Vec<_>, UnionError>` and propagate failure via
        // `.ok()?` so a SpecTag/FlowSignal disagreement in the locals
        // projection aborts the merge rather than silently None-
        // killing the offending slot.
        let locals_w: Vec<Option<crate::flowspace::model::Hlvalue>> = (0..len)
            .map(|i| {
                let w1 = self_view.get(i).and_then(|c| c.as_ref());
                let w2 = other_view.get(i).and_then(|c| c.as_ref());
                crate::flowspace::framestate::union(w1, w2)
            })
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        drop(self_view);
        drop(other_view);
        // Derive `entries` from `locals_w` — `locals_w` is now the
        // primary carrier matching `framestate.py:19 self.locals_w`,
        // and `entries` is a backward-compatibility view for callers
        // that have not yet migrated.  `Hlvalue::Variable(v)` is
        // cloned directly into the entry, carrying the merged value's
        // identity (`framestate.py:108/113-114 union` returns either a
        // carry-through Variable or a fresh phi `Variable()`).
        // `Hlvalue::Constant(_)` cells carry no Variable identity
        // (Constants are not Variable-bound in the pyre IR), so they
        // collapse to `None` in the `entries` view — matching the
        // prior Variable-only locals fold's lossy projection on the
        // few sites where a Constant could land in locals via
        // `setstate`.
        let entries: Vec<Option<crate::flowspace::model::Variable>> = locals_w
            .iter()
            .map(|slot| {
                slot.as_ref().and_then(|hv| match hv {
                    crate::flowspace::model::Hlvalue::Variable(v) => Some(v.clone()),
                    crate::flowspace::model::Hlvalue::Constant(_) => None,
                })
            })
            .collect();
        Some(FrameState {
            entries,
            locals_w,
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
    pub fn getoutputargs(&self, target: &FrameState, graph: &FunctionGraph) -> Vec<LinkArg> {
        self.try_getoutputargs(target, graph).expect(
            "getoutputargs: target Variable slot unbound in self / stack-length mismatch \
             — union invariant violated",
        )
    }

    /// Fallible [`Self::getoutputargs`].  Returns `None` instead of
    /// panicking when a target `Variable` slot has no corresponding
    /// `Some` cell in `self` (locals) or no positional cell in `self`'s
    /// flattened stack.  For union-derived target states this never
    /// happens (union None-kills a slot unless every predecessor binds
    /// it), so [`Self::getoutputargs`] keeps its panicking contract; the
    /// cyclic framestate path pre-seeds loop-header entries with live-in
    /// phis that bypass the union, so it threads links through this
    /// checked variant and declines a phantom-slot mismatch to the
    /// monotonic fallback.
    pub fn try_getoutputargs(
        &self,
        target: &FrameState,
        graph: &FunctionGraph,
    ) -> Option<Vec<LinkArg>> {
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
        // Upstream returns `List[Hlvalue]` (Variable | Constant cells).
        // Pyre's matching IR carrier is `LinkArg`
        // (`Value(Variable) | Const(Constant)`) — a closed sum that
        // aligns 1:1 with the upstream cell domain.  Hlvalue→LinkArg
        // routing per cell is centralised in `hlvalue_to_linkarg`:
        //   - `Hlvalue::Variable(v)` → `LinkArg::Value(v.clone())` —
        //     the Variable identity carries through inline, so a
        //     single upstream `Variable` reaches every Link threading
        //     it without a separate slot allocation.
        //   - `Hlvalue::Constant(c)` → `LinkArg::Const(c.clone())` —
        //     direct carry, no synthetic op or slot required.
        //
        // Upstream's `mergeable` is `locals_w + recursively_flatten(
        // stack) + [exc_type, exc_value]`.  Pyre walks the same three
        // projections in the same order so the positional mapping
        // between target and self is preserved across the locals→
        // stack→exception boundary.
        //
        // Locals projection consults `locals_w` (the `Hlvalue` carrier
        // matching `framestate.py:19 self.locals_w`) for both the
        // target's Variable predicate and self's cell contribution —
        // same shape as the stack and exception projections below.
        // `locals_w_view` returns the populated `Hlvalue` slice for
        // union/getstate-derived states and derives it from `entries`
        // for hand-built fixtures that pre-date the Z4.A.6 swap.
        let mut result: Vec<LinkArg> = Vec::new();
        // (1) locals projection — `framestate.mergeable` head.
        let target_locals_view = target.locals_w_view(graph);
        let self_locals_view = self.locals_w_view(graph);
        for (i, w_target) in target_locals_view.iter().enumerate() {
            if matches!(
                w_target,
                Some(crate::flowspace::model::Hlvalue::Variable(_))
            ) {
                let w_self = self_locals_view.get(i).and_then(|c| c.as_ref())?;
                result.push(hlvalue_to_linkarg(w_self));
            }
        }
        // (2) stack projection — `recursively_flatten(stack)` middle
        // segment.  Walk both flattened stacks in step so position-`i`
        // lines up between the target and self mergeable views; push
        // `self_flat_stack[i]` (routed through `hlvalue_to_linkarg`)
        // at every position where the target cell is a `Variable`.
        let target_flat_stack = crate::flowspace::framestate::recursively_flatten(&target.stack);
        let self_flat_stack = crate::flowspace::framestate::recursively_flatten(&self.stack);
        for (i, w_target) in target_flat_stack.iter().enumerate() {
            if matches!(w_target, crate::flowspace::model::Hlvalue::Variable(_)) {
                let w_self = self_flat_stack.get(i)?;
                result.push(hlvalue_to_linkarg(w_self));
            }
        }
        // (3) exception args projection — `[exc_type, exc_value]`
        // tail per `framestate.py:34-39 mergeable`.  `exc_args`
        // substitutes `Constant(None)` sentinels when no exception is
        // pending, so the Variable predicate skips empty-exception
        // states; non-empty exception cells get the same Hlvalue→
        // LinkArg routing as the stack projection.
        let target_exc = exc_args(&target.last_exception);
        let self_exc = exc_args(&self.last_exception);
        for (w_target, w_self) in target_exc.iter().zip(self_exc.iter()) {
            if matches!(w_target, crate::flowspace::model::Hlvalue::Variable(_)) {
                result.push(hlvalue_to_linkarg(w_self));
            }
        }
        Some(result)
    }

    /// Enumerate every `Variable` cell across the full mergeable
    /// projection — RPython `framestate.py:50-51 getvariables` parity:
    ///
    /// ```python
    /// def getvariables(self):
    ///     return [w for w in self.mergeable if isinstance(w, Variable)]
    /// ```
    ///
    /// Walks `locals + recursively_flatten(stack) + [exc_type, exc_value]`
    /// in that order — the same `mergeable` shape `getoutputargs` traverses
    /// — and returns every Variable cell in positional order.  Used at
    /// merge-block construction (`SpamBlock(framestate)` upstream) so the
    /// block's `inputargs` line up 1:1 with `getoutputargs(target=self)`
    /// from any predecessor.
    ///
    /// Locals projection consults `self.locals_w` per upstream
    /// `framestate.py:50-51 getvariables`'s `mergeable` walk —
    /// `mergeable`'s locals head IS `locals_w`.  `locals_w_view`
    /// returns the populated `Hlvalue` slice for union/getstate-derived
    /// states and derives it from `entries` for hand-built fixtures
    /// that pre-date the Z4.A.6 swap.  Stack and exception
    /// projections carry `Hlvalue` directly and route through the
    /// same Variable filter as upstream.
    pub fn getvariables(&self, graph: &FunctionGraph) -> Vec<crate::flowspace::model::Variable> {
        use crate::flowspace::model::Hlvalue;
        let mut out: Vec<crate::flowspace::model::Variable> = Vec::new();
        // (1) Locals — `framestate.mergeable` head.
        for slot in self.locals_w_view(graph).iter() {
            if let Some(Hlvalue::Variable(v)) = slot {
                out.push(v.clone());
            }
        }
        // (2) Stack — `recursively_flatten(self.stack)` middle segment.
        for h in crate::flowspace::framestate::recursively_flatten(&self.stack) {
            if let Hlvalue::Variable(v) = h {
                out.push(v);
            }
        }
        // (3) Exception args — `[exc_type, exc_value]` tail.
        for h in exc_args(&self.last_exception).iter() {
            if let Hlvalue::Variable(v) = h {
                out.push(v.clone());
            }
        }
        out
    }

    /// `framestate.py:42-48 FrameState.copy` — "Make a copy of this
    /// state in which all Variables are fresh."
    ///
    /// ```python
    /// def copy(self):
    ///     exc = self.last_exception
    ///     if exc is not None:
    ///         exc = FSException(_copy(exc.w_type), _copy(exc.w_value))
    ///     return FrameState(map(_copy, self.locals_w),
    ///                       map(_copy, self.stack),
    ///                       exc, self.blocklist, self.next_offset)
    /// ```
    ///
    /// `_copy(v)` (framestate.py:4-12) independently creates a fresh
    /// `Variable(v)` per occurrence — NO shared mapping across cells.
    /// Constants and None slots pass through unchanged.
    pub fn copy(&self, graph: &mut FunctionGraph) -> FrameState {
        use crate::flowspace::model::{Hlvalue, Variable};
        // `_copy(v)` — each call independently mints a fresh Variable.
        // `Variable(v)` (model.py:300-311) copies the source's name
        // prefix via `rename(v)`, which reads `v._name`; carry the
        // source slot's name onto the fresh Variable so the freshened
        // state's phi inputs keep their human-readable names.
        let copy_hlvalue = |h: &Hlvalue, graph: &mut FunctionGraph| -> Hlvalue {
            match h {
                Hlvalue::Variable(src) => {
                    // `_copy(v) -> Variable(v)` (framestate.py:4): fresh
                    // Variable carrying the source's `_name` prefix
                    // (annotation stays None).
                    let mut nv = graph.alloc_value_var();
                    nv.rename_from(src);
                    Hlvalue::Variable(nv)
                }
                other => other.clone(),
            }
        };
        let copy_var = |v: &Variable, graph: &mut FunctionGraph| -> Variable {
            let mut nv = graph.alloc_value_var();
            nv.rename_from(v);
            nv
        };
        // `locals_w` is the authoritative carrier (framestate.py:19).
        // When `locals_w` is empty (pre-Z4.A.6 fixtures), fall back
        // to `entries` as the seed.
        let locals_w: Vec<Option<Hlvalue>> = if self.locals_w.len() == self.entries.len() {
            self.locals_w
                .iter()
                .map(|slot| slot.as_ref().map(|h| copy_hlvalue(h, graph)))
                .collect()
        } else {
            self.entries
                .iter()
                .map(|e| e.as_ref().map(|v| Hlvalue::Variable(copy_var(v, graph))))
                .collect()
        };
        let entries: Vec<Option<Variable>> = locals_w
            .iter()
            .map(|slot| {
                slot.as_ref().and_then(|hv| match hv {
                    Hlvalue::Variable(v) => Some(v.clone()),
                    Hlvalue::Constant(_) => None,
                })
            })
            .collect();
        // `map(_copy, self.stack)` — each stack element independently copied.
        // framestate.py:8-10: FlowSignal → rebuild with recursively
        // copied args.
        let stack: Vec<crate::flowspace::framestate::StackElem> =
            self.stack
                .iter()
                .map(|elem| match elem {
                    crate::flowspace::framestate::StackElem::Value(h) => {
                        crate::flowspace::framestate::StackElem::Value(copy_hlvalue(h, graph))
                    }
                    crate::flowspace::framestate::StackElem::Signal(sig) => {
                        use crate::flowspace::flowcontext::FlowSignal;
                        let copied_args: Vec<Hlvalue> =
                            sig.args().iter().map(|a| copy_hlvalue(a, graph)).collect();
                        crate::flowspace::framestate::StackElem::Signal(
                            FlowSignal::rebuild_with_args(sig.tag(), copied_args),
                        )
                    }
                })
                .collect();
        // `exc = FSException(_copy(exc.w_type), _copy(exc.w_value))`
        let last_exception = self.last_exception.as_ref().map(|exc| {
            crate::flowspace::model::FSException::new(
                copy_hlvalue(&exc.w_type, graph),
                copy_hlvalue(&exc.w_value, graph),
            )
        });
        FrameState {
            entries,
            locals_w,
            stack,
            last_exception,
            blocklist: self.blocklist.clone(),
            next_offset: self.next_offset,
        }
    }

    /// `framestate.py:53-64 FrameState.matches` — "Two states match if
    /// they only differ by using different Variables at the same place."
    ///
    /// ```python
    /// def matches(self, other):
    ///     assert self.blocklist == other.blocklist
    ///     assert self.next_offset == other.next_offset
    ///     for w1, w2 in zip(self.mergeable, other.mergeable):
    ///         if not (w1 == w2 or (isinstance(w1, Variable) and
    ///                              isinstance(w2, Variable))):
    ///             return False
    ///     return True
    /// ```
    ///
    /// blocklist and next_offset must agree (asserted) — `matches` is a
    /// post-`union` comparison invoked from `flowcontext.py:438` and the
    /// candidate-list precondition is that all candidates share the
    /// same join-point coordinates.
    ///
    /// The mergeable projection walks `locals + recursively_flatten(
    /// stack) + [exc_type, exc_value]` (same shape as `getvariables` /
    /// `getoutputargs`).  Two cells match iff they are equal or both
    /// are `Variable` cells (regardless of identity — phi slot equality
    /// is by structure, not Variable identity).  `(None, None)` matches; any other
    /// mix involving `None` does not.
    pub fn matches(&self, other: &FrameState, graph: &FunctionGraph) -> bool {
        use crate::flowspace::model::Hlvalue;
        assert_eq!(
            self.blocklist, other.blocklist,
            "matches: blocklist mismatch"
        );
        assert_eq!(
            self.next_offset, other.next_offset,
            "matches: next_offset mismatch"
        );
        // (1) Locals — `framestate.mergeable` head.
        let self_locals = self.locals_w_view(graph);
        let other_locals = other.locals_w_view(graph);
        if self_locals.len() != other_locals.len() {
            return false;
        }
        let opt_cell_match = |w1: &Option<Hlvalue>, w2: &Option<Hlvalue>| match (w1, w2) {
            (None, None) => true,
            (Some(Hlvalue::Variable(_)), Some(Hlvalue::Variable(_))) => true,
            (Some(x), Some(y)) => x == y,
            _ => false,
        };
        if !self_locals
            .iter()
            .zip(other_locals.iter())
            .all(|(w1, w2)| opt_cell_match(w1, w2))
        {
            return false;
        }
        drop(self_locals);
        drop(other_locals);
        // (2) Stack — `recursively_flatten(self.stack)` middle segment.
        let self_stack = crate::flowspace::framestate::recursively_flatten(&self.stack);
        let other_stack = crate::flowspace::framestate::recursively_flatten(&other.stack);
        if self_stack.len() != other_stack.len() {
            return false;
        }
        let cell_match = |w1: &Hlvalue, w2: &Hlvalue| match (w1, w2) {
            (Hlvalue::Variable(_), Hlvalue::Variable(_)) => true,
            (x, y) => x == y,
        };
        if !self_stack
            .iter()
            .zip(other_stack.iter())
            .all(|(w1, w2)| cell_match(w1, w2))
        {
            return false;
        }
        // (3) Exception args — `[exc_type, exc_value]` tail.
        let self_exc = exc_args(&self.last_exception);
        let other_exc = exc_args(&other.last_exception);
        self_exc
            .iter()
            .zip(other_exc.iter())
            .all(|(w1, w2)| cell_match(w1, w2))
    }
}

/// Hlvalue→LinkArg routing per cell, matching upstream
/// `framestate.py:92-99 getoutputargs` which appends the polymorphic
/// `mergeable[i]` cell directly into `Link.args`.  Pyre's `LinkArg` is
/// the matching closed sum: `Hlvalue::Variable(v)` → `LinkArg::Value(v)`
/// (the Variable carries its identity inline so downstream readers
/// match on the Variable object directly);
/// `Hlvalue::Constant(c)` → `LinkArg::Const(c)` (direct carry, no
/// synthetic op or slot allocation required for the Constant domain).
fn hlvalue_to_linkarg(w: &crate::flowspace::model::Hlvalue) -> LinkArg {
    use crate::flowspace::model::Hlvalue;
    match w {
        Hlvalue::Variable(v) => LinkArg::Value(v.clone()),
        Hlvalue::Constant(c) => LinkArg::Const(c.clone()),
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
                let target_inputargs: Vec<crate::flowspace::model::Variable> =
                    target_block.inputargs.clone();
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
                let subst: HashMap<crate::flowspace::model::Variable, LinkArg> = target_inputargs
                    .into_iter()
                    .zip(link_args.into_iter())
                    .collect();
                // upstream: `link.args = [v.replace(subst) for v in
                // exit.args]`.
                let new_args: Vec<LinkArg> = target_exit
                    .args
                    .iter()
                    .map(|arg| match arg.as_variable() {
                        Some(v) => subst.get(v).cloned().unwrap_or_else(|| arg.clone()),
                        None => arg.clone(),
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

/// RPython `remove_assertion_errors(graph)` (simplify.py:321-346) over
/// the front model graph.
///
/// ```python
/// def remove_assertion_errors(graph):
///     """Remove branches that go directly to raising an AssertionError,
///     assuming that AssertionError shouldn't occur at run-time.  Note that
///     this is how implicit exceptions are removed (see _implicit_ in
///     flowcontext.py).
///     """
///     for block in list(graph.iterblocks()):
///         for i in range(len(block.exits)-1, -1, -1):
///             exit = block.exits[i]
///             if not (exit.target is graph.exceptblock and
///                     exit.args[0] == Constant(AssertionError)):
///                 continue
///             if len(block.exits) < 2:
///                 break
///             if block.canraise:
///                 if exit.exitcase is None:
///                     break
///                 if len(block.exits) == 2:
///                     block.exitswitch = None
///                     exit.exitcase = None
///             lst = list(block.exits)
///             del lst[i]
///             block.recloseblock(*lst)
/// ```
///
/// One extension beyond the upstream body: when the removal leaves a
/// single exit on a *non-canraise value switch*, the survivor is
/// promoted to an unconditional link (`block.exitswitch = None;
/// exits[0].exitcase = None` — `backendopt/removeassert.py:84-89
/// kill_assertion_link`).  Upstream flow graphs only carry Bool or
/// last-exception exitswitches into this pass, and `flatten.py
/// insert_exits` tolerates a leftover single Bool case (`assert
/// link.exitcase in (None, False, True)`); Rust enum-match lowering
/// produces Int discriminant switches whose leftover `Int` case that
/// assert rejects, so the `kill_assertion_link` normalisation applies
/// here.  For the canraise two-exit case the survivor (`exits[0]`)
/// already has `exitcase = None`, so the one code path covers both.
///
/// Returns the number of removed exits so the caller can gate the
/// follow-up dead-condition sweep (`removeassert.py:35-37` — "now melt
/// away the (hopefully) dead operation that compute the condition").
pub fn remove_assertion_errors(graph: &mut FunctionGraph) -> usize {
    use crate::flowspace::model::HOST_ENV;
    let assert_err_class = HOST_ENV
        .lookup_builtin("AssertionError")
        .expect("HOST_ENV missing AssertionError");
    let exceptblock = graph.exceptblock;
    let mut removed = 0usize;
    // upstream: `for block in list(graph.iterblocks())`.
    for block_idx in 0..graph.blocks.len() {
        let mut i = graph.blocks[block_idx].exits.len();
        while i > 0 {
            i -= 1;
            let block = &graph.blocks[block_idx];
            let Some(exit) = block.exits.get(i) else {
                break;
            };
            // upstream: `if not (exit.target is graph.exceptblock and
            // exit.args[0] == Constant(AssertionError)): continue`.
            let targets_except = exit.target == exceptblock;
            let args_is_assert_err = matches!(
                exit.args.first(),
                Some(LinkArg::Const(c))
                    if matches!(
                        &c.value,
                        ConstValue::HostObject(h) if *h == assert_err_class
                    )
            );
            if !(targets_except && args_is_assert_err) {
                continue;
            }
            // upstream: `if len(block.exits) < 2: break`.
            if block.exits.len() < 2 {
                break;
            }
            // upstream: `if block.canraise: if exit.exitcase is None:
            // break`.
            if block.canraise() && exit.exitcase.is_none() {
                break;
            }
            let exits_len = block.exits.len();
            let block = &mut graph.blocks[block_idx];
            // upstream: `lst = list(block.exits); del lst[i];
            // block.recloseblock(*lst)`.
            block.exits.remove(i);
            if exits_len == 2 {
                // Promote the survivor to an unconditional link —
                // upstream's canraise arm (`simplify.py:333-335`) plus
                // the `kill_assertion_link` normalisation for value
                // switches (`removeassert.py:84-89`, see above).  Clear
                // the low-level case too so no branch metadata lingers on
                // a now-unconditional edge, matching `fold_constant_exitswitch`.
                block.exitswitch = None;
                block.exits[0].exitcase = None;
                block.exits[0].llexitcase = None;
            }
            removed += 1;
        }
    }
    removed
}

/// Fold a constant exitswitch to its taken link — the model-layer
/// slice of `constant_fold_graph`'s link folding
/// (`rpython/translator/backendopt/constfold.py`): when a block's
/// `exitswitch` Variable is bound in the same block by a constant
/// (`ConstBool` / `ConstInt`, optionally through the `bool` UnaryOp
/// wrap [`FunctionGraph::set_branch`] appends), keep only the
/// matching exit and promote it to an unconditional link.  The dead
/// condition ops melt away in the caller's [`prune_dead_phis`]
/// sweep; the disconnected arm is emptied by
/// [`clear_unreachable_blocks`].  Returns the number of folded
/// switches.
pub fn fold_constant_exitswitch(graph: &mut FunctionGraph) -> usize {
    let mut folded = 0usize;
    for block_idx in 0..graph.blocks.len() {
        let block = &graph.blocks[block_idx];
        let Some(ExitSwitch::Value(sw)) = block.exitswitch.clone() else {
            continue;
        };
        let def_of = |v: &crate::flowspace::model::Variable| {
            block
                .operations
                .iter()
                .find(|op| op.result.as_ref() == Some(v))
                .map(|op| &op.kind)
        };
        let mut kind = def_of(&sw);
        if let Some(OpKind::UnaryOp { op, operand, .. }) = kind
            && op == "bool"
        {
            kind = def_of(operand);
        }
        // The matched exitcase spellings: an `If` branch carries
        // `ExitCase::Bool` (`set_branch`), a MIR `SwitchInt` carries
        // `ExitCase::Const(Int)` plus the `"default"` catch-all arm
        // (`front::mir` terminator lowering).
        let (bool_case, int_case) = match kind {
            Some(OpKind::ConstBool(b)) => (Some(*b), Some(i64::from(*b))),
            Some(OpKind::ConstInt(n)) => ((*n == 0 || *n == 1).then(|| *n != 0), Some(*n)),
            _ => continue,
        };
        let matches_case = |link: &Link| match &link.exitcase {
            Some(ExitCase::Bool(b)) => Some(*b) == bool_case,
            Some(ExitCase::Const(ConstValue::Int(n))) => Some(*n) == int_case,
            _ => false,
        };
        let is_default = |link: &Link| {
            matches!(
                &link.exitcase,
                Some(ExitCase::Const(ConstValue::UniStr(s))) if s == "default"
            )
        };
        // `replace_exitswitch_by_constant` (simplify.py:36-48) collects
        // *every* arm whose exitcase equals the constant and asserts exactly
        // one survivor (`assert len(newexits) == 1`), falling back to the
        // `"default"` catch-all only when none match: a constant matching
        // two arms is a malformed switch that checkgraph's exitcase
        // uniqueness invariant (`flowspace/model.py:686`) forbids.
        let matching: Vec<usize> = block
            .exits
            .iter()
            .enumerate()
            .filter(|(_, link)| matches_case(link))
            .map(|(idx, _)| idx)
            .collect();
        let chosen = match matching.as_slice() {
            [only] => *only,
            [] => {
                let Some(default_idx) = block.exits.iter().position(is_default) else {
                    continue;
                };
                default_idx
            }
            _ => panic!(
                "const-fold: constant switch matched {} arms (duplicate exitcase) in graph {:?}",
                matching.len(),
                graph.name
            ),
        };
        let block = &mut graph.blocks[block_idx];
        let mut taken = block.exits.swap_remove(chosen);
        taken.exitcase = None;
        taken.llexitcase = None;
        block.exits = vec![taken];
        block.exitswitch = None;
        folded += 1;
    }
    folded
}

/// Empty every block unreachable from `graph.startblock` in place —
/// operations, exits, inputargs cleared, the `Vec` slot kept because
/// `BlockId` doubles as the index.  Needed after
/// [`fold_constant_exitswitch`] disconnects an arm: the registry
/// lift (`translate_op`) and [`prune_dead_phis`] both walk blocks by
/// index, and `prune_dead_phis` pins no-predecessor blocks as extra
/// entry points, so a disconnected arm would otherwise keep its dead
/// ops alive.
pub fn clear_unreachable_blocks(graph: &mut FunctionGraph) {
    let mut reachable = vec![false; graph.blocks.len()];
    let mut worklist = vec![graph.startblock];
    while let Some(b) = worklist.pop() {
        if std::mem::replace(&mut reachable[b.0], true) {
            continue;
        }
        for link in &graph.blocks[b.0].exits {
            worklist.push(link.target);
        }
    }
    // The return / except sinks stay even when no live link reaches
    // them — graph plumbing addresses them unconditionally.
    reachable[graph.returnblock.0] = true;
    reachable[graph.exceptblock.0] = true;
    for (idx, block) in graph.blocks.iter_mut().enumerate() {
        if !reachable[idx] {
            block.operations.clear();
            block.exits.clear();
            block.inputargs.clear();
            block.exitswitch = None;
            // Drop the locals snapshot too: a retained framestate keeps
            // variable references alive into later analysis passes, which
            // defeats the unreachable-block drain.
            block.framestate = None;
        }
    }
}

/// Remove dead aggregate constructions — a `SyntheticTransparentCtor`
/// (or a no-arg `Box::new_uninit`) call whose result escapes *only* into
/// its own `FieldWrite` field stores (never read, returned, or passed) —
/// along with those stores.
///
/// This is the `remove_simple_mallocs` shape from
/// `rpython/translator/backendopt/malloc.py`: a `malloc` whose result
/// flows only into `setfield` stores and is never read is dead, so the
/// malloc and its stores are dropped.  `prune_dead_phis`
/// (`transform_dead_op_vars`) cannot do this on its own because a
/// `FieldWrite` is side-effecting (not a pure op), so it pins its
/// `base` operand in `read_vars` and the aggregate survives.
///
/// The construction only reaches the front-end as a malloc + store
/// chain (`OpKind::Call { SyntheticTransparentCtor }` +
/// `OpKind::FieldWrite`) for the heterogeneous tuple/struct path; the
/// pure `OpKind::NewTuple` form is already swept by `prune_dead_phis`.
/// A discarded `(a, b)` pair (e.g. `let _ = (truth, expect_true);`)
/// lowers to this chain, and its dead field stores otherwise force a
/// primitive→object store whose `(BoolRepr|IntegerRepr, InstanceRepr)`
/// convert the rtyper rejects.
///
/// Conservative by construction: a constructor result is treated as
/// dead only when its *every* use is as a `FieldWrite.base` (the one
/// read site this pass exempts) and every such store has no result of
/// its own.  Any other reference — a `FieldRead`/`InteriorFieldWrite`
/// base, a `FieldWrite` *value*, a call argument, an exitswitch, a
/// `Link.arg`, a terminal-block inputarg — pins the result and the
/// chain is kept.  Runs to a fixpoint so a store of one dead aggregate
/// into another exposes the inner one once the outer store is gone.
pub fn remove_dead_aggregates(graph: &mut FunctionGraph) -> usize {
    use crate::flowspace::model::Variable;

    let is_removable_ctor = |kind: &OpKind| match kind {
        OpKind::Call {
            target: CallTarget::SyntheticTransparentCtor { .. },
            ..
        } => true,
        // `Box::new_uninit()` — a no-arg heap-box allocation. The `vec![…]`
        // construction (`box_assume_init_into_vec_unsafe(box [..])`) whose
        // consumer `front::mir` rewrote to `newlist` leaves the box dead:
        // its result flows only into the `FieldWrite` that stored the now-
        // unused `Array` aggregate. Treat it as a removable ctor so the box,
        // its store, and (once un-pinned) the `Array` cascade out together.
        OpKind::Call {
            target: CallTarget::FunctionPath { segments },
            args,
            ..
        } => {
            let tail = ["boxed", "Box", "new_uninit"];
            args.is_empty()
                && segments.len() >= tail.len()
                && segments[segments.len() - tail.len()..]
                    .iter()
                    .zip(tail.iter())
                    .all(|(s, t)| s == t)
        }
        _ => false,
    };

    let mut total_removed = 0usize;
    loop {
        // Pass 1: gather every "real" read.  A `FieldWrite.base` is the
        // store *target*, not a value read, so it is the single ref this
        // pass does not count; `FieldWrite.value` (if a Variable) is a
        // genuine read and every other op contributes its full
        // `op_variable_refs` set.
        let mut real_read: HashSet<Variable> = HashSet::new();
        for block in &graph.blocks {
            for op in &block.operations {
                if let OpKind::FieldWrite { value, .. } = &op.kind {
                    if let Some(var) = value.as_variable() {
                        real_read.insert(var.clone());
                    }
                } else {
                    for var in crate::inline::op_variable_refs(&op.kind) {
                        real_read.insert(var);
                    }
                }
            }
            match &block.exitswitch {
                Some(ExitSwitch::Value(var)) => {
                    real_read.insert(var.clone());
                }
                // A fused comparison reads its operands at the branch.
                Some(ExitSwitch::Fused { args, .. }) => {
                    for arg in args {
                        real_read.insert(arg.clone());
                    }
                }
                Some(ExitSwitch::LastException) | None => {}
            }
            for link in &block.exits {
                for arg in &link.args {
                    if let Some(var) = arg.as_variable() {
                        real_read.insert(var.clone());
                    }
                }
            }
            // A terminal block (no exits) implicitly uses every inputarg
            // — the same return-value pin `prune_dead_phis` Step 1 makes.
            if block.exits.is_empty() {
                for iarg in &block.inputargs {
                    real_read.insert(iarg.clone());
                }
            }
        }

        // Pass 2: a constructor result is dead when it is not read and
        // every store into it is result-less (so dropping the store
        // leaves no dangling definition).
        let mut dead: HashSet<Variable> = HashSet::new();
        for block in &graph.blocks {
            for op in &block.operations {
                if !is_removable_ctor(&op.kind) {
                    continue;
                }
                let Some(result) = &op.result else { continue };
                if real_read.contains(result) {
                    continue;
                }
                let stores_clean =
                    graph
                        .blocks
                        .iter()
                        .flat_map(|b| &b.operations)
                        .all(|o| match &o.kind {
                            OpKind::FieldWrite { base, .. } if base == result => o.result.is_none(),
                            _ => true,
                        });
                if stores_clean {
                    dead.insert(result.clone());
                }
            }
        }
        if dead.is_empty() {
            break;
        }

        // Pass 3: drop the dead constructors and their field stores.
        let mut removed = 0usize;
        for block in &mut graph.blocks {
            block.operations.retain(|op| {
                let drop = if is_removable_ctor(&op.kind) {
                    op.result.as_ref().is_some_and(|r| dead.contains(r))
                } else if let OpKind::FieldWrite { base, .. } = &op.kind {
                    dead.contains(base)
                } else {
                    false
                };
                if drop {
                    removed += 1;
                }
                !drop
            });
        }
        total_removed += removed;
        if removed == 0 {
            break;
        }
    }
    total_removed
}

/// Fuse the boxing-constructor idiom into a native GC allocation.
///
/// pyre's boxing constructors (`floatobject::w_float_new` etc.) are written
/// in Rust as `malloc_typed(W_FloatObject { ob_header: …, floatval: v })` —
/// construct the whole struct on the stack, then heap-copy.  `front::mir`
/// lowers that to a `SyntheticTransparentCtor` aggregate (`%agg`) + per-field
/// `FieldWrite`s + a residual `Call(pyre_object::lltype::malloc_typed, [%agg])`.
/// RPython's orthodox form is alloc-then-init (`p = malloc(S); p.f = v`); the
/// rtyper lowers `malloc` to the GC allocation op.  pyre's rtyper is an
/// ephemeral type oracle that never rewrites the surviving model graph, so the
/// lowering is produced here instead: rewrite the cluster to
/// `NewWithVtable { owner, vtable } -> %ret` + a single payload
/// `FieldWrite(%ret, …)`, dropping the `ob_header` (PyObject base) subtree.
/// The type pointer the dropped `ob_header.ob_type` store carries is captured
/// into `NewWithVtable.vtable` (the runtime stamps the new object's `ob_type` /
/// `w_class` from it — `type_id` resolves struct size only), matching the
/// runtime tracer oracle (`box_trace.rs trace_box_float`: one `NewWithVtable`
/// carrying a real type pointer + one `SetfieldGc` for the payload).
///
/// The payload store is inserted *after* the `NewWithVtable` (which reuses the
/// malloc result Variable `%ret`), since the original aggregate field stores
/// precede the malloc call and would be use-before-def if retargeted in place.
/// The orphaned aggregate ctor + header `FieldWrite`s become dead and are swept
/// by the `remove_dead_aggregates` + `prune_dead_phis` passes that follow in
/// `simplify_lowered_graph`.
pub fn fuse_boxing_alloc(graph: &mut FunctionGraph) -> usize {
    use crate::flowspace::model::Variable;
    // Recognised boxing structs and their scalar payload fields, in struct
    // order.  The header (`ob_header`: ob_type + w_class) is NOT listed as a
    // payload — its type pointer is captured separately into
    // `NewWithVtable.vtable` (see `resolve_vtable_addr`) and the runtime stamps
    // `ob_type` / `w_class` from it, so only the scalar payload setfield(s) are
    // re-emitted (oracle: `box_trace.rs trace_box_float` / `trace_box_int`).
    fn payload_fields(owner: &str) -> Option<&'static [(&'static str, ValueType)]> {
        match owner {
            "W_FloatObject" => Some(&[("floatval", ValueType::Float)]),
            "W_IntObject" => Some(&[("intval", ValueType::Int)]),
            "W_ComplexObject" => Some(&[("real", ValueType::Float), ("imag", ValueType::Float)]),
            // `value: *mut BigInt` — a raw pointer payload, stored as a ref-kind
            // setfield (`Ref(None)`, opaque pointee).
            "W_LongObject" => Some(&[("value", ValueType::Ref(None))]),
            _ => None,
        }
    }

    let is_malloc_typed = |target: &CallTarget| -> bool {
        matches!(target, CallTarget::FunctionPath { segments }
            if segments.len() >= 2
                && segments[segments.len() - 1] == "malloc_typed"
                && segments[segments.len() - 2] == "lltype")
    };

    // Resolve the type-pointer the dropped `ob_header.ob_type` store carries:
    // `%agg.ob_header = %h; %h.ob_type = __pyre_cast_instance(ConstRefAddr(t))`.
    // The runtime stamps the new object's `ob_type`/`w_class` from this address
    // (read out of the `NewWithVtable` size descriptor), so it must travel with
    // the op rather than being dropped.  Returns `0` when the cluster carries no
    // resolvable constant type-pointer (e.g. a synthetic test fixture).
    fn store_value(graph: &FunctionGraph, base: &Variable, field_name: &str) -> Option<Variable> {
        graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .find_map(|o| match &o.kind {
                OpKind::FieldWrite {
                    base: b,
                    field,
                    value,
                    ..
                } if b == base && field.name.as_str() == field_name => value.as_variable().cloned(),
                _ => None,
            })
    }
    fn const_ref_addr(graph: &FunctionGraph, var: &Variable, depth: u32) -> Option<i64> {
        if depth == 0 {
            return None;
        }
        let producer = graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .find(|o| o.result.as_ref() == Some(var))?;
        match &producer.kind {
            OpKind::ConstRefAddr(addr) => Some(*addr),
            // Walk `__pyre_cast_instance[<root>]` pointer reinterprets.
            OpKind::Call {
                target: CallTarget::FunctionPath { segments },
                args,
                ..
            } if segments.first().map(String::as_str) == Some("__pyre_cast_instance")
                && args.len() == 1 =>
            {
                const_ref_addr(graph, &args[0], depth - 1)
            }
            _ => None,
        }
    }
    let resolve_vtable_addr = |graph: &FunctionGraph, agg: &Variable| -> i64 {
        store_value(graph, agg, "ob_header")
            .and_then(|header| store_value(graph, &header, "ob_type"))
            .and_then(|obtype| const_ref_addr(graph, &obtype, 8))
            .unwrap_or(0)
    };

    struct Payload {
        field: FieldDescriptor,
        value: LinkArg,
        ty: ValueType,
    }
    struct Site {
        block: usize,
        op: usize,
        result: crate::flowspace::model::Variable,
        owner: String,
        vtable: i64,
        payloads: Vec<Payload>,
    }

    let mut sites: Vec<Site> = Vec::new();
    for (bi, block) in graph.blocks.iter().enumerate() {
        for (oi, op) in block.operations.iter().enumerate() {
            let OpKind::Call { target, args, .. } = &op.kind else {
                continue;
            };
            if !is_malloc_typed(target) || args.len() != 1 {
                continue;
            }
            let Some(result) = &op.result else { continue };
            let agg = &args[0];
            // `%agg` must be a `SyntheticTransparentCtor` for a known boxing
            // struct.  Search graph-wide: the ctor and the malloc call land in
            // the same block, but the field stores feeding the aggregate may sit
            // in earlier blocks (each preceding call ends a block).
            let owner = graph
                .blocks
                .iter()
                .flat_map(|b| &b.operations)
                .find_map(|o| match (&o.result, &o.kind) {
                    (
                        Some(r),
                        OpKind::Call {
                            target: CallTarget::SyntheticTransparentCtor { name, .. },
                            ..
                        },
                    ) if r == agg => Some(name.clone()),
                    _ => None,
                });
            let Some(owner) = owner else { continue };
            let Some(fields) = payload_fields(&owner) else {
                continue;
            };
            // Resolve every payload field's store: `FieldWrite { base: %agg,
            // field.name == payload }`.  A malformed cluster missing any payload
            // store is left untouched so the annotate wall still flags it rather
            // than emitting a half-initialised allocation.
            let mut payloads = Vec::with_capacity(fields.len());
            let mut complete = true;
            for &(field_name, ref payload_ty) in fields {
                let found = graph
                    .blocks
                    .iter()
                    .flat_map(|b| &b.operations)
                    .find_map(|o| match &o.kind {
                        OpKind::FieldWrite {
                            base, field, value, ..
                        } if base == agg && field.name.as_str() == field_name => {
                            Some((field.clone(), value.clone()))
                        }
                        _ => None,
                    });
                match found {
                    Some((field, value)) => payloads.push(Payload {
                        field,
                        value,
                        ty: payload_ty.clone(),
                    }),
                    None => {
                        complete = false;
                        break;
                    }
                }
            }
            if !complete {
                continue;
            }
            let vtable = resolve_vtable_addr(graph, agg);
            sites.push(Site {
                block: bi,
                op: oi,
                result: result.clone(),
                owner,
                vtable,
                payloads,
            });
        }
    }

    let fused = sites.len();
    // Rewrite in reverse (block, op) order so the per-site `insert` does not
    // shift the indices of not-yet-processed sites in the same block.
    for site in sites.into_iter().rev() {
        let block = &mut graph.blocks[site.block];
        block.operations[site.op] = SpaceOperation {
            result: Some(site.result.clone()),
            kind: OpKind::NewWithVtable {
                owner: site.owner,
                vtable: site.vtable,
            },
        };
        // Payload stores follow the `NewWithVtable`, in struct order.  Each is
        // a plain `FieldWrite` the assembler lowers to its own `setfield_gc`.
        for (k, payload) in site.payloads.into_iter().enumerate() {
            block.operations.insert(
                site.op + 1 + k,
                SpaceOperation {
                    result: None,
                    kind: OpKind::FieldWrite {
                        base: site.result.clone(),
                        field: payload.field,
                        value: payload.value,
                        ty: payload.ty,
                    },
                },
            );
        }
    }
    if fused > 0 {
        prune_dead_boxing_remnants(graph);
    }
    fused
}

/// Re-thread op operands the boxing lowering left referenced across a block
/// boundary without a matching inputarg.  [`fuse_boxing_alloc`] relocates the
/// payload `FieldWrite` to the `malloc_typed` site and the dead-var sweeps
/// (`prune_dead_boxing_remnants` / `prune_dead_phis`) then strip the inputargs
/// that carried the boxing cluster's cross-block values — the scalar payload
/// (a parameter defined before the `get_instantiate` call that ends the entry
/// block) and the `__pyre_cast_instance` chain that turns the `NewWithVtable`
/// result into the returned `PyObjectRef`.  This pass runs *after* those
/// sweeps and re-threads every such operand through the predecessor chain,
/// restoring the per-block operand invariant the adapter
/// (`flowspace_adapter::function_graph_to_flowspace`) requires — every
/// referenced operand defined as a block inputarg or op result.
///
/// An operand is threaded only when its definition is reachable from the use
/// block through a chain of single-predecessor blocks — i.e. the definition
/// dominates the use along a strictly linear path, exactly the shape the
/// boxing cluster's `alloc → cast → cast → return` chain has.  This is the
/// precondition under which [`FunctionGraph::ensure_variable_at_block`] threads
/// cleanly: with one predecessor at every step, every path into the use block
/// passes through the definition, so no predecessor edge is left unable to
/// supply the value.  An operand reaching a join or loop block (multiple
/// predecessors), or one with no upstream definition at all, is left untouched
/// — the adapter still Skips that graph, and the no-definition panic is never
/// reached.  Graphs with no cross-block-undefined operand collect no work and
/// stay byte-identical.
pub fn thread_undefined_op_operands(graph: &mut FunctionGraph) {
    use crate::flowspace::model::Variable;
    use std::collections::{HashMap, HashSet};

    // target block id → source block ids feeding it.
    let mut preds: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for b in &graph.blocks {
        for e in &b.exits {
            preds.entry(e.target).or_default().push(b.id);
        }
    }

    // `var` is defined in a block reachable from `block` by walking
    // single-predecessor edges only.  Mirrors the success condition of
    // `ensure_variable_at_block` for a linear predecessor chain.
    fn defined_via_single_pred_chain(
        graph: &FunctionGraph,
        preds: &HashMap<BlockId, Vec<BlockId>>,
        block: BlockId,
        var: &Variable,
    ) -> bool {
        let mut cur = block;
        let mut seen: HashSet<BlockId> = HashSet::new();
        loop {
            if !seen.insert(cur) {
                return false;
            }
            let Some(ps) = preds.get(&cur) else {
                return false;
            };
            if ps.len() != 1 {
                return false;
            }
            let p = ps[0];
            if graph.variable_defined_in_block(p, var) {
                return true;
            }
            cur = p;
        }
    }

    let mut work: Vec<(BlockId, Variable)> = Vec::new();
    for block in &graph.blocks {
        if block.dead {
            continue;
        }
        for op in &block.operations {
            for var in crate::inline::op_variable_refs(&op.kind) {
                if !graph.variable_defined_in_block(block.id, &var)
                    && defined_via_single_pred_chain(graph, &preds, block.id, &var)
                {
                    work.push((block.id, var));
                }
            }
        }
    }
    // `ensure_variable_at_block` is idempotent, so a duplicate (block, var)
    // pair from two operands referencing the same Variable is a harmless no-op
    // on the second call.
    for (block_id, var) in work {
        graph.ensure_variable_at_block(block_id, &var);
    }
}

/// Sweep the construct-on-stack header remnants that [`fuse_boxing_alloc`]
/// orphans.  Once `malloc_typed(struct)` becomes a `NewWithVtable` (which
/// carries the type pointer / `w_class` through its vtable descriptor) the
/// original aggregate ctor, the inner `PyObject` header ctor, their
/// `ob_header` / `ob_type` / `w_class` field stores, and the
/// `__pyre_cast_instance` casts feeding `ob_type` / `w_class` are all dead.
///
/// `front::mir` threads each of those dead values across the block boundary
/// the preceding `Call` opens (`get_instantiate` / `malloc_typed` each end a
/// block) by reusing the *same* `Variable` as both the producing block's op
/// result and the successor block's inputarg.  Neither generic sweep reclaims
/// the cluster in that shape:
///
///   * [`remove_dead_aggregates`] (and the old form of this sweep) counts
///     every `Link.arg` as an unconditional read, so the threaded copy keeps
///     the producer alive even though nothing on the far side reads it.
///   * [`prune_dead_phis`] keeps the producer because a `FieldWrite` is
///     side-effecting and pins its `base`.
///
/// So this sweep runs `transform_dead_op_vars`' dependency-flow liveness
/// (`simplify.py:425-479`) — a `Link.arg` is live iff the target inputarg it
/// feeds is live, not unconditionally — combined with the malloc-removal
/// exemption (`malloc.py remove_simple_mallocs`): a store into a fresh
/// aggregate nothing reads is itself dead, so a `SyntheticTransparentCtor`
/// `FieldWrite.base` is *not* a liveness root.  The exemption is scoped to
/// ctor bases — a store *through* an aliasing or loaded base (a cast or a
/// load) is a real heap side effect and roots its base like any other op.
/// The header producers eligible for removal are all side-effect-free:
/// a `SyntheticTransparentCtor` stack construct, a `__pyre_cast_instance`
/// pointer reinterpret (`exception_cannot_occur` → `cast_pointer`), and the
/// `pyre_object::pyobject::get_instantiate` read of a type's `instantiate`
/// slot feeding the dropped `w_class`.  `get_instantiate` is an `Acquire`
/// atomic load of an init-once slot (the `set_instantiate` `Release` mutator
/// writes it during `init_typeobjects`); it is removable not because the slot
/// is immutable but because the load is observation-free — dropping a load
/// whose result no reader reaches only relaxes ordering nothing depends on.
/// Their operands route through `dependencies[result]` like a pure op rather than
/// pinning — so a dead cast drops its constant feed and a dead `get_instantiate`
/// drops the cast feeding it.  Liveness still gates removal, so a producer the
/// flow reaches (e.g. a `get_instantiate` whose result is genuinely used) is
/// never removed.  One global liveness pass reaches the whole cross-block
/// cluster at once; the inputargs / link args the removed producers leave
/// dangling (and the now-dead address constants) are reclaimed by the
/// [`prune_dead_phis`] pass the lowering runs next.
pub(crate) fn prune_dead_boxing_remnants(graph: &mut FunctionGraph) {
    use crate::flowspace::model::Variable;
    use std::collections::HashMap;

    let is_removable_producer = |kind: &OpKind| match kind {
        OpKind::Call {
            target: CallTarget::SyntheticTransparentCtor { .. },
            ..
        } => true,
        OpKind::Call {
            target: CallTarget::FunctionPath { segments },
            args,
            ..
        } => {
            // `__pyre_cast_instance[<root>]` — the front-end pointer-downcast
            // narrow (`front::mir`), always a single-operand reinterpret.
            // Pin the arity so an unrelated multi-arg path that happens to
            // share the synthetic marker leaf is never swept as a cast.
            let is_cast = segments.first().map(String::as_str) == Some("__pyre_cast_instance")
                && args.len() == 1;
            // `pyre_object::pyobject::get_instantiate` — the pure
            // `instantiate`-slot read feeding the dropped `w_class`.  Match
            // the full owner path rather than the bare leaf so a future
            // side-effecting function sharing the `get_instantiate` name in
            // some other module can never be classified removable.
            let get_instantiate = ["pyre_object", "pyobject", "get_instantiate"];
            let is_get_instantiate = segments.len() >= get_instantiate.len()
                && segments[segments.len() - get_instantiate.len()..]
                    .iter()
                    .map(String::as_str)
                    .eq(get_instantiate.iter().copied());
            is_cast || is_get_instantiate
        }
        _ => false,
    };

    let block_index: HashMap<BlockId, usize> = graph
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect();

    // Fresh stack aggregates (`SyntheticTransparentCtor`) are the only bases
    // whose field stores may be dropped: a store into an aggregate nothing
    // reads is itself dead (`malloc.py remove_simple_mallocs`).  A store
    // *through* an aliasing or loaded base (`__pyre_cast_instance` /
    // `get_instantiate` / a parameter / …) is a real heap side effect, so the
    // exemption is scoped to ctor results — every other store roots its base.
    let synthetic_ctor_results: HashSet<Variable> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { .. },
                    ..
                }
            )
        })
        .filter_map(|op| op.result.clone())
        .collect();

    // Liveness roots + dependency edges (`simplify.py:425-462`) with the two
    // boxing exemptions above.
    let mut read_vars: HashSet<Variable> = HashSet::new();
    let mut dependencies: HashMap<Variable, Vec<Variable>> = HashMap::new();
    for block in &graph.blocks {
        for op in &block.operations {
            match &op.kind {
                // A store into a fresh aggregate is live iff the aggregate is
                // live; the `base` is deliberately not rooted, so an unread
                // aggregate's stores die with it (malloc-removal exemption).
                OpKind::FieldWrite { base, value, .. } if synthetic_ctor_results.contains(base) => {
                    if let Some(var) = value.as_variable() {
                        dependencies
                            .entry(base.clone())
                            .or_default()
                            .push(var.clone());
                    }
                }
                // A store through any other base is a real heap side effect:
                // root both the base and the stored value.
                OpKind::FieldWrite { base, value, .. } => {
                    read_vars.insert(base.clone());
                    if let Some(var) = value.as_variable() {
                        read_vars.insert(var.clone());
                    }
                }
                // Side-effect-free header producers route operands like a pure
                // op, so a dead producer drops its feed instead of pinning it.
                kind if is_removable_producer(kind) => match &op.result {
                    Some(result) => dependencies
                        .entry(result.clone())
                        .or_default()
                        .extend(crate::inline::op_variable_refs(kind)),
                    None => read_vars.extend(crate::inline::op_variable_refs(kind)),
                },
                // Every other op is side-effecting: its operands are real reads.
                kind => read_vars.extend(crate::inline::op_variable_refs(kind)),
            }
        }
        match &block.exitswitch {
            Some(ExitSwitch::Value(var)) => {
                read_vars.insert(var.clone());
            }
            // A fused comparison reads its operands at the branch.
            Some(ExitSwitch::Fused { args, .. }) => {
                for arg in args {
                    read_vars.insert(arg.clone());
                }
            }
            Some(ExitSwitch::LastException) | None => {}
        }
        // Terminal blocks implicitly read every inputarg (`simplify.py:459-462`).
        if block.exits.is_empty() {
            read_vars.extend(block.inputargs.iter().cloned());
        }
        // Cross-block: a link arg is live iff the target inputarg it feeds is.
        for link in &block.exits {
            let Some(&ti) = block_index.get(&link.target) else {
                continue;
            };
            let target_iargs = &graph.blocks[ti].inputargs;
            for (arg, target_iarg) in link.args.iter().zip(target_iargs.iter()) {
                if let Some(arg_var) = arg.as_variable() {
                    dependencies
                        .entry(target_iarg.clone())
                        .or_default()
                        .push(arg_var.clone());
                }
            }
        }
    }
    // Real parameters are always live (`simplify.py:431-433` start inputargs).
    if let Some(&i) = block_index.get(&graph.startblock) {
        read_vars.extend(graph.blocks[i].inputargs.iter().cloned());
    }
    // Backward flow (`simplify.py:471-479`).
    let mut pending: Vec<Variable> = read_vars.iter().cloned().collect();
    while let Some(var) = pending.pop() {
        if let Some(deps) = dependencies.get(&var).cloned() {
            for dep in deps {
                if read_vars.insert(dep.clone()) {
                    pending.push(dep);
                }
            }
        }
    }

    // A removable producer whose result the flow leaves unread is dead.  A
    // ctor is eligible only when its every store is result-less (dropping it
    // leaves no dangling definition), matching `remove_dead_aggregates`.
    let dead: HashSet<Variable> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter_map(|op| {
            let result = op.result.as_ref()?;
            if read_vars.contains(result) || !is_removable_producer(&op.kind) {
                return None;
            }
            let stores_clean =
                graph
                    .blocks
                    .iter()
                    .flat_map(|b| &b.operations)
                    .all(|o| match &o.kind {
                        OpKind::FieldWrite { base, .. } if base == result => o.result.is_none(),
                        _ => true,
                    });
            stores_clean.then(|| result.clone())
        })
        .collect();
    if dead.is_empty() {
        return;
    }

    // Drop the dead producers and every field store targeting one.
    for block in &mut graph.blocks {
        block.operations.retain(|op| {
            if let Some(r) = &op.result {
                if dead.contains(r) {
                    return false;
                }
            }
            if let OpKind::FieldWrite { base, .. } = &op.kind {
                if dead.contains(base) {
                    return false;
                }
            }
            true
        });
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
/// TODO: `start_blocks` is `{graph.startblock} ∪
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
///      result `Variable` in `read_vars`; the dead ones are swept here
///      alongside Step 7's inputarg trim (the matching
///      `block.inputargs[i]` removal becomes a no-op when the
///      Input op is already gone).  Naked `OpKind::Input` ops
///      (legacy frontend fallback for cross-block reads not yet
///      routed through the lazy installer) are likewise removed
///      when their result Variable is dead — `simplify.py` itself has
///      no standalone Input shape, but the line-by-line port
///      sweeps them under the same canremove rule.
///
///      TODO: translator-gated arms not ported.
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
///      Variable (`simplify.py:520-524`).
pub fn prune_dead_phis(graph: &mut FunctionGraph) {
    use crate::inline::is_pure_op;
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
    // A no-predecessor block is a legitimate calling-convention entry
    // only when every inputarg is a genuine parameter — the result of
    // an `OpKind::Input` op in that same block (the closure-entry shape
    // exercised by `prune_dead_phis_skips_non_canonical_entry_blocks`).
    // jtransform can leave *unreachable* merge blocks whose inputargs
    // are phi targets referencing values defined in reachable blocks; a
    // phi with no predecessor to fill it is malformed, and pinning such
    // a block as an entry would keep its dead operands (and any value
    // sharing their register) alive into regalloc.  Restrict the
    // orphan-entry roots to genuine parameter blocks so dead merge
    // blocks are excluded.
    let is_genuine_entry = |block: &Block| -> bool {
        block.inputargs.iter().all(|iarg| {
            block.operations.iter().any(|op| {
                matches!(op.kind, OpKind::Input { .. }) && op.result.as_ref() == Some(iarg)
            })
        })
    };
    let start_blocks: HashSet<BlockId> = std::iter::once(start)
        .chain(
            graph
                .blocks
                .iter()
                .filter(|b| {
                    b.id != start && !with_predecessor.contains(&b.id) && is_genuine_entry(b)
                })
                .map(|b| b.id),
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
    // even if their result Variable is unread — the raise side-effect
    // is observable.
    // RPython `simplify.py:425-426`:
    //     read_vars = set()  # set of variables really used
    //     dependencies = defaultdict(set) # map {Var: list-of-Vars-it-depends-on}
    //
    // Both sidetables are Variable-keyed (PyPy uses Python object
    // identity for Variable's __hash__ / __eq__); pyre matches via
    // Rc<i64>-backed Variable._nr.
    let mut read_vars: HashSet<crate::flowspace::model::Variable> = HashSet::new();
    let mut dependencies: HashMap<
        crate::flowspace::model::Variable,
        Vec<crate::flowspace::model::Variable>,
    > = HashMap::new();
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
            let operands: Vec<crate::flowspace::model::Variable> =
                crate::inline::op_variable_refs(&op.kind);
            // `simplify.py:441-445`:
            //   if not canremove(op, block):    read_vars.update(args)
            //   else:                           dependencies[result] += args
            let removable = is_pure_op(&op.kind) && Some(i) != raising_op_idx;
            if let Some(result_var) = op.result.clone()
                && removable
            {
                dependencies.entry(result_var).or_default().extend(operands);
            } else {
                for var in operands {
                    read_vars.insert(var);
                }
            }
        }
        match &block.exitswitch {
            Some(ExitSwitch::Value(var)) => {
                read_vars.insert(var.clone());
            }
            // A fused comparison reads its operands at the branch.
            Some(ExitSwitch::Fused { args, .. }) => {
                for arg in args {
                    read_vars.insert(arg.clone());
                }
            }
            Some(ExitSwitch::LastException) | None => {}
        }
        // `simplify.py:459-462`: terminal blocks (no exits)
        // implicitly use every inputarg.
        if block.exits.is_empty() {
            for iarg in &block.inputargs {
                read_vars.insert(iarg.clone());
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
            for (arg, target_iarg) in link.args.iter().zip(target_block.inputargs.iter()) {
                if let Some(arg_var) = arg.as_variable() {
                    dependencies
                        .entry(target_iarg.clone())
                        .or_default()
                        .push(arg_var.clone());
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
            for iarg in &graph.blocks[i].inputargs {
                read_vars.insert(iarg.clone());
            }
        }
    }

    // Step 4: backward flow.
    // `simplify.py:471-479 flow_read_var_backward`.
    let mut pending: Vec<crate::flowspace::model::Variable> = read_vars.iter().cloned().collect();
    while let Some(var) = pending.pop() {
        if let Some(deps) = dependencies.get(&var).cloned() {
            for dep in deps {
                if read_vars.insert(dep.clone()) {
                    pending.push(dep);
                }
            }
        }
    }

    // Step 5: drop every pure op whose result is dead.
    // `simplify.py:484-488 if op.result not in read_vars: if
    // canremove(op, block): del block.operations[i]`.
    // Inputarg-shaped Input ops keep their result Variable in `read_vars`
    // via Step 1+3+dependency-routing, so the live ones survive
    // unconditionally; dead ones get removed alongside their
    // matching inputarg trim in Step 7.  Naked Input ops (any Input
    // op whose result is not pinned by a matching inputarg) do
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
    // Pre-compute the (block_id, op_index) pairs of dead pure ops in
    // an immutable pass; the mutable removal pass below operates on
    // the precomputed list to avoid an overlapping borrow.
    //
    // RPython `simplify.py:484-488`:
    //     if op.result not in read_vars:
    //         if canremove(op, block):
    //             del block.operations[i]
    let mut dead_op_positions: Vec<(BlockId, usize)> = Vec::new();
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
            let dead = match op.result.as_ref() {
                Some(r) => !read_vars.contains(r),
                None => false,
            };
            if dead && is_pure_op(&op.kind) && Some(i) != raising_op_idx {
                dead_op_positions.push((block.id, i));
            }
        }
    }
    for block in &mut graph.blocks {
        let mut dead_indices: Vec<usize> = dead_op_positions
            .iter()
            .filter_map(|(bid, idx)| (*bid == block.id).then_some(*idx))
            .collect();
        dead_indices.sort_unstable();
        for i in dead_indices.into_iter().rev() {
            block.operations.remove(i);
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
            let target_iargs: Vec<crate::flowspace::model::Variable> = {
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
                let target_iarg = &target_iargs[i];
                if !read_vars.contains(target_iarg) {
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
        let inputargs: Vec<crate::flowspace::model::Variable> =
            graph.blocks[block_idx].inputargs.clone();
        for i in (0..inputargs.len()).rev() {
            let iarg = &inputargs[i];
            if read_vars.contains(iarg) {
                continue;
            }
            graph.blocks[block_idx].inputargs.remove(i);
            if let Some(op_idx) = graph.blocks[block_idx].operations.iter().position(|op| {
                matches!(op.kind, OpKind::Input { .. }) && op.result.as_ref() == Some(iarg)
            }) {
                graph.blocks[block_idx].operations.remove(op_idx);
            }
        }
    }

    remove_duplicate_inputargs(graph);
}

/// Crate-local [`FunctionGraph`] port of RPython
/// `translator/simplify.py:540-590 remove_identical_vars_SSA`.
///
/// The earlier subset only removed literally duplicated `inputargs`, which
/// missed upstream's phi-tuple equivalence pass and left codewriter graphs
/// with self-conflicting register-allocation inputs.  Keep the same shape as
/// upstream: collect each block's incoming phi columns, union equivalent
/// variables, rebuild input/link arg lists, then rename the remaining block
/// body references.
pub fn remove_duplicate_inputargs(graph: &mut FunctionGraph) {
    use crate::tool::algo::unionfind::{UnionFind, UnionFindInfo};
    use std::collections::HashMap;

    let start = graph.startblock;
    let return_block = graph.returnblock;
    let except_block = graph.exceptblock;

    #[derive(Clone)]
    struct Representative {
        rep: LinkArg,
    }

    impl UnionFindInfo for Representative {
        fn absorb(&mut self, _other: Self) {}
    }

    let mut uf: UnionFind<LinkArg, Representative> =
        UnionFind::new(|arg: &LinkArg| Representative { rep: arg.clone() });

    let mut entries: HashMap<BlockId, Vec<(usize, usize)>> = HashMap::new();
    for (pred_idx, block) in graph.blocks.iter().enumerate() {
        for (link_idx, link) in block.exits.iter().enumerate() {
            if link.target == start || link.target == return_block || link.target == except_block {
                continue;
            }
            entries
                .entry(link.target)
                .or_default()
                .push((pred_idx, link_idx));
        }
    }

    let mut inputs: HashMap<BlockId, Vec<(crate::flowspace::model::Variable, Vec<LinkArg>)>> =
        HashMap::new();
    for (&block_id, links) in &entries {
        let inputargs = graph.block(block_id).inputargs.clone();
        let mut phis = Vec::with_capacity(inputargs.len());
        for (arg_i, input) in inputargs.into_iter().enumerate() {
            let mut phi_args = Vec::with_capacity(links.len());
            for (pred_idx, link_idx) in links {
                let link = &graph.blocks[*pred_idx].exits[*link_idx];
                let arg = link.args.get(arg_i).unwrap_or_else(|| {
                    panic!(
                        "remove_identical_vars_SSA: link.args[{arg_i}] missing \
                         (graph {}, target {:?})",
                        graph.name, block_id,
                    )
                });
                phi_args.push(arg.clone());
            }
            phis.push((input, phi_args));
        }
        inputs.insert(block_id, phis);
    }

    fn simplify_phis(
        uf: &mut UnionFind<LinkArg, Representative>,
        phis: &mut Vec<(crate::flowspace::model::Variable, Vec<LinkArg>)>,
    ) -> bool {
        let mut to_remove: Vec<usize> = Vec::new();
        let mut unique_phis: HashMap<Vec<LinkArg>, crate::flowspace::model::Variable> =
            HashMap::new();
        for (i, (input, phi_args)) in phis.iter().enumerate() {
            let new_args: Vec<LinkArg> = phi_args
                .iter()
                .map(|arg| uf.find_rep(arg.clone()))
                .collect();
            // PRE-EXISTING-ADAPTATION: the all-equal phi collapse of
            // `simplify.py:561-563` (`if all_equal(new_args):
            // uf.union(new_args[0], input)`) is omitted here.  Upstream that
            // collapse deliberately produces cross-block variable references
            // — it removes a merge block's inputarg and leaves the body
            // reading a value defined in a predecessor — and relies on the
            // next `all_passes` entry, `SSA_to_SSI` (backendopt/ssa.py:135-196),
            // to repair them by re-threading every used-but-undefined variable
            // as a fresh inputarg through all incoming links.  pyre runs this
            // pass on the `crate::model` front-end graph, which has no
            // `SSA_to_SSI` (the faithful `ssa_to_ssi` port at
            // translator/backendopt/ssa.rs operates on `crate::flowspace::model`,
            // a distinct IR).  Without the repair the collapse strands the body
            // reference and the flowspace adapter rejects it as an undefined
            // operand.  `prune_dead_phis` (transform_dead_op_vars) has already
            // dropped every dead inputarg, so each surviving column is used and
            // `SSA_to_SSI` would re-thread it regardless — the collapse is a
            // no-op net of the repair, and skipping it yields the same graph
            // with the column left threaded.  The codewriter's phi-tuple
            // equivalence (`simplify.py:565-568`) is the duplicate-column merge
            // handled by `unique_phis` below, which is SSI-safe because both
            // columns are inputargs of the same block.  Convergence path: port
            // `SSA_to_SSI` to `crate::model` (or unify the two `FunctionGraph`
            // IRs and run the standard `all_passes` including `ssa_to_ssi`).
            if let Some(existing) = unique_phis.get(&new_args).cloned() {
                uf.union(LinkArg::Value(existing), LinkArg::Value(input.clone()));
                to_remove.push(i);
            } else {
                unique_phis.insert(new_args, input.clone());
            }
        }
        for i in to_remove.iter().rev() {
            phis.remove(*i);
        }
        !to_remove.is_empty()
    }

    let block_ids: Vec<BlockId> = inputs.keys().copied().collect();
    let mut progress = true;
    while progress {
        progress = false;
        for block_id in &block_ids {
            if simplify_phis(&mut uf, inputs.get_mut(block_id).expect("inputs block")) {
                progress = true;
            }
        }
    }

    let keys: Vec<LinkArg> = uf.keys().cloned().collect();
    let mut renaming: HashMap<
        crate::flowspace::model::Variable,
        crate::flowspace::model::Variable,
    > = HashMap::new();
    for key in keys {
        let LinkArg::Value(src) = key.clone() else {
            continue;
        };
        if let Some(info) = uf.get(&key) {
            if let LinkArg::Value(dst) = &info.rep {
                if src != *dst {
                    renaming.insert(src, dst.clone());
                }
            }
        }
    }

    for (&block_id, phis) in &inputs {
        let surviving: Vec<crate::flowspace::model::Variable> =
            phis.iter().map(|(input, _)| input.clone()).collect();
        {
            // simplify.py drops the phi slot from `block.inputargs` and the
            // predecessor `link.args`.  Pyre phi blocks also carry a matching
            // `OpKind::Input` op per slot, so remove the Input ops whose result
            // is a dropped slot — otherwise the later rename leaves a spurious
            // in-block definition for a value no predecessor supplies.
            let surviving_set: std::collections::HashSet<&crate::flowspace::model::Variable> =
                surviving.iter().collect();
            let block = graph.block_mut(block_id);
            let dropped: Vec<crate::flowspace::model::Variable> = block
                .inputargs
                .iter()
                .filter(|v| !surviving_set.contains(*v))
                .cloned()
                .collect();
            block.inputargs = surviving;
            for slot in &dropped {
                if let Some(op_idx) = block.operations.iter().position(|op| {
                    matches!(op.kind, OpKind::Input { .. }) && op.result.as_ref() == Some(slot)
                }) {
                    block.operations.remove(op_idx);
                }
            }
        }
        let links = entries
            .get(&block_id)
            .expect("entry list for input block")
            .clone();
        for (link_pos, (pred_idx, link_idx)) in links.into_iter().enumerate() {
            graph.blocks[pred_idx].exits[link_idx].args = phis
                .iter()
                .map(|(_, phi_args)| phi_args[link_pos].clone())
                .collect();
        }
    }

    if renaming.is_empty() {
        return;
    }

    let remap_var = |var: &crate::flowspace::model::Variable| {
        renaming.get(var).cloned().unwrap_or_else(|| var.clone())
    };
    for block in &mut graph.blocks {
        for inputarg in &mut block.inputargs {
            *inputarg = remap_var(inputarg);
        }
        for op in &mut block.operations {
            op.result = op.result.as_ref().map(&remap_var);
            op.kind = crate::inline::remap_op_kind(&op.kind, &remap_var);
        }
        let (exitswitch, exits) = remap_control_flow_metadata_var(
            &block.exitswitch,
            &block.exits,
            &remap_var,
            |block_id| block_id,
        );
        block.exitswitch = exitswitch;
        block.exits = exits;
    }

    // A second sweep removes any same-Variable inputargs created by the final
    // rename.  This is the same fixed-point effect as upstream's loop.
    for block_idx in 0..graph.blocks.len() {
        let block_id = graph.blocks[block_idx].id;
        if block_id == start || block_id == return_block || block_id == except_block {
            continue;
        }
        let mut first_seen: HashMap<crate::flowspace::model::Variable, usize> = HashMap::new();
        let mut duplicate_slots: Vec<(usize, usize)> = Vec::new();
        for (i, inputarg) in graph.blocks[block_idx].inputargs.iter().enumerate() {
            if let Some(first_i) = first_seen.get(inputarg).copied() {
                duplicate_slots.push((i, first_i));
            } else {
                first_seen.insert(inputarg.clone(), i);
            }
        }
        if duplicate_slots.is_empty() {
            continue;
        }
        let mut removable_slots: Vec<(usize, usize)> = Vec::new();
        for (dup_i, first_i) in duplicate_slots {
            let mut removable = true;
            for pred_idx in 0..graph.blocks.len() {
                for link in &graph.blocks[pred_idx].exits {
                    if link.target != block_id {
                        continue;
                    }
                    if link.args.get(dup_i) != link.args.get(first_i) {
                        removable = false;
                        break;
                    }
                }
                if !removable {
                    break;
                }
            }
            if removable {
                removable_slots.push((dup_i, first_i));
            }
        }
        for (dup_i, _) in removable_slots.into_iter().rev() {
            // Drop the `OpKind::Input` op paired with the DUPLICATE slot by its
            // slot position — `del block.inputargs[i]` by index
            // (simplify.py:650-653).  After the rename several Input ops can
            // share the survivor's result, so a result match would risk
            // dropping the survivor (first-seen) slot's own `name`/`ty`/
            // `class_root` metadata instead of the duplicate's.  The k-th
            // `OpKind::Input` op mirrors the k-th inputarg (paired in order at
            // construction; earlier passes preserve that order).
            let mut seen_inputs = 0usize;
            let mut dup_op_idx = None;
            for (op_idx, op) in graph.blocks[block_idx].operations.iter().enumerate() {
                if matches!(op.kind, OpKind::Input { .. }) {
                    if seen_inputs == dup_i {
                        dup_op_idx = Some(op_idx);
                        break;
                    }
                    seen_inputs += 1;
                }
            }
            graph.blocks[block_idx].inputargs.remove(dup_i);
            if let Some(op_idx) = dup_op_idx {
                graph.blocks[block_idx].operations.remove(op_idx);
            }
            for pred_idx in 0..graph.blocks.len() {
                for link in &mut graph.blocks[pred_idx].exits {
                    if link.target == block_id {
                        link.args.remove(dup_i);
                    }
                }
            }
        }
    }
}

impl Block {
    pub fn canraise(&self) -> bool {
        matches!(self.exitswitch, Some(ExitSwitch::LastException))
    }

    /// Variable-identity iter over [`Self::inputargs`] — direct
    /// over the upstream-orthodox `Vec<Variable>` storage, matching
    /// `Block.inputargs: List[Variable]` (`flowspace/model.py:21-25`).
    pub fn input_variables(&self) -> impl Iterator<Item = &crate::flowspace::model::Variable> + '_ {
        self.inputargs.iter()
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
/// `link.args` — Variable-direct.  Takes a `Fn(&Variable) -> Variable`
/// renamer so callers whose alias map is keyed on Variable identity
/// (e.g. jtransform's `Transformer.aliases: HashMap<Variable,
/// Variable>` or inline's per-block value_map) write the rename
/// directly without round-tripping through a slot index.  Mirrors
/// upstream's Variable-identity rewrite path.
pub fn remap_control_flow_metadata_var<FValue, FBlock>(
    exitswitch: &Option<ExitSwitch>,
    exits: &[Link],
    remap_value: FValue,
    remap_block: FBlock,
) -> (Option<ExitSwitch>, Vec<Link>)
where
    FValue: Fn(&crate::flowspace::model::Variable) -> crate::flowspace::model::Variable,
    FBlock: Fn(BlockId) -> BlockId,
{
    let remap_link_arg = |arg: &LinkArg| -> LinkArg {
        match arg {
            LinkArg::Value(var) => LinkArg::Value(remap_value(var)),
            LinkArg::Const(value) => LinkArg::Const(value.clone()),
        }
    };
    let exitswitch = exitswitch.as_ref().map(|switch| match switch {
        ExitSwitch::Value(var) => ExitSwitch::Value(remap_value(var)),
        ExitSwitch::LastException => ExitSwitch::LastException,
        ExitSwitch::Fused { opname, args } => ExitSwitch::Fused {
            opname: opname.clone(),
            args: args.iter().map(&remap_value).collect(),
        },
    });
    let exits = exits
        .iter()
        .map(|link| Link {
            args: link.args.iter().map(&remap_link_arg).collect(),
            target: remap_block(link.target),
            exitcase: link.exitcase.clone(),
            prevblock: link.prevblock.map(&remap_block),
            llexitcase: link.llexitcase.clone(),
            last_exception: link.last_exception.as_ref().map(&remap_link_arg),
            last_exc_value: link.last_exc_value.as_ref().map(&remap_link_arg),
        })
        .collect();
    (exitswitch, exits)
}

/// `Variable.concretetype` analogue collapsed to the four kinds the
/// codewriter needs.
///
/// RPython stores `.concretetype` inline on each `Variable` after
/// `RPythonTyper.specialize()` rewrites the graph; pyre reads the same
/// attribute off the `Variable` directly, so
/// [`FunctionGraph::concretetype_of`] reads
/// `getkind(var.concretetype.borrow())` line-for-line with upstream.
/// `Unknown` is the pre-rtyper sentinel — `Variable` analogue before
/// the rtyper ran.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    use crate::translator::rtyper::lltypesystem::lltype::{GcKind, LowLevelType};
    match ty {
        // `if TYPE is lltype.Void: return "void"`
        LowLevelType::Void => ConcreteType::Void,
        // `if TYPE is lltype.Float and supports_floats: return 'float'`
        LowLevelType::Float => ConcreteType::Float,
        // `if TYPE is lltype.SingleFloat and supports_singlefloats:
        //      return 'int'  # singlefloats are stored in an int`
        LowLevelType::SingleFloat => ConcreteType::Signed,
        // `if rffi.sizeof(TYPE) > rffi.sizeof(lltype.Signed):
        //      if supports_longlong and TYPE is not lltype.LongFloat:
        //          assert rffi.sizeof(TYPE) == 8
        //          return 'float'`
        // — target-size dependent: on 64-bit `Signed` is 8 bytes so
        // SignedLongLong (also 8) does NOT exceed it and falls through
        // to `return "int"`.  On 32-bit `Signed` is 4 bytes so the
        // 8-byte longlong variants take the `'float'` slot.  Pyre's
        // host word size is `usize` / `isize`, mirrored via
        // `std::mem::size_of`.
        LowLevelType::SignedLongLong | LowLevelType::UnsignedLongLong => {
            if std::mem::size_of::<i64>() > std::mem::size_of::<isize>() {
                ConcreteType::Float
            } else {
                ConcreteType::Signed
            }
        }
        // Other Primitives → `"int"`.  Includes Signed, Unsigned,
        // Bool, Char, UniChar, Address.  RPython's
        // `rffi.sizeof(TYPE) > rffi.sizeof(lltype.Signed)` check
        // only fires for the longlong family handled above and the
        // 16-byte longlonglong family handled below.
        LowLevelType::Signed
        | LowLevelType::Unsigned
        | LowLevelType::Bool
        | LowLevelType::Char
        | LowLevelType::UniChar
        | LowLevelType::Address => ConcreteType::Signed,
        // `if rffi.sizeof(TYPE) > rffi.sizeof(lltype.Signed) and not supports_longlong
        //      or TYPE is lltype.LongFloat:
        //      raise NotImplementedError("type %s is too large" % TYPE)`
        // — pyre panics with the same shape until the longlonglong /
        // longfloat backends land; production paths never reach this.
        LowLevelType::SignedLongLongLong
        | LowLevelType::UnsignedLongLongLong
        | LowLevelType::LongFloat => {
            panic!("getkind: type {ty:?} not supported (history.py:62 NotImplementedError)")
        }
        // `elif isinstance(TYPE, lltype.Ptr):
        //      if TYPE.TO._gckind == 'raw': return "int"
        //      else: return "ref"`
        LowLevelType::Ptr(ptr) => match ptr_gckind(&ptr.TO) {
            GcKind::Raw => ConcreteType::Signed,
            GcKind::Gc | GcKind::Prebuilt => ConcreteType::GcRef,
        },
        // RPython does not place `InteriorPtr` directly on a
        // Variable's `concretetype`; it reaches the JIT codewriter
        // only as the source of a `getinteriorfield` op.  Reaching
        // this arm means the rtyper handed the codewriter an
        // unsupported shape (same `NotImplementedError` family as
        // upstream `history.py:62,70`).  Panic with the canonical
        // `getkind: …not supported…` payload so
        // [`crate::translator::rtyper::cutover::lowleveltype_to_concrete`]
        // can route this to a fail-loud `TyperError::missing_rtype_operation`
        // instead of silently coercing the operand to `GcRef`.
        LowLevelType::InteriorPtr(_) => {
            panic!("getkind: type {ty:?} not supported as concretetype (history.py:70)")
        }
        // RPython `Func`/`Struct`/`Array`/`FixedSizeArray`/`Opaque`/
        // `ForwardReference` are not valid `concretetype` values for
        // a Variable — they only appear as the `TO` of a `Ptr`.
        // Reaching this arm means the rtyper handed the codewriter
        // a non-pointer aggregate, which would `NotImplementedError`
        // upstream.
        LowLevelType::Func(_)
        | LowLevelType::Struct(_)
        | LowLevelType::Array(_)
        | LowLevelType::FixedSizeArray(_)
        | LowLevelType::Opaque(_)
        | LowLevelType::ForwardReference(_) => {
            panic!("getkind: type {ty:?} not supported as concretetype (history.py:70)")
        }
    }
}

/// Project a [`ConcreteType`] back to a canonical
/// `LowLevelType` representative — the inverse direction of
/// [`getkind`] for the kind-only adapter callers (regalloc canonical
/// exceptblock, type_state synth merge, codewriter test
/// scaffolding).  Each branch chooses a representative that
/// round-trips through `getkind`:
///
/// - `Signed` → `LowLevelType::Signed`.
/// - `Float`  → `LowLevelType::Float`.
/// - `Void`   → `LowLevelType::Void`.
/// - `GcRef`  → [`crate::translator::rtyper::rclass::OBJECTPTR`]
///   (`Ptr(OBJECT)`, the canonical GC pointer the rtyper itself
///   stamps onto exception-block `evalue` Variables).
/// - `Unknown` → `None` (Variable.concretetype stays unset;
///   [`FunctionGraph::concretetype`] reads `Unknown` for the slot).
///
/// Lossy for richer Ptr-of-specific-Struct values, so
/// [`FunctionGraph::set_concretetype`] only writes through when the
/// existing Variable type's `getkind` does NOT already match the
/// requested kind (i.e. the rtyper's authoritative type wins).
pub(crate) fn concrete_to_canonical_lltype(
    ty: ConcreteType,
) -> Option<crate::translator::rtyper::lltypesystem::lltype::LowLevelType> {
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    match ty {
        ConcreteType::Signed => Some(LowLevelType::Signed),
        ConcreteType::Float => Some(LowLevelType::Float),
        ConcreteType::Void => Some(LowLevelType::Void),
        ConcreteType::GcRef => Some(crate::translator::rtyper::rclass::OBJECTPTR.clone()),
        ConcreteType::Unknown => None,
    }
}

/// Helper for [`getkind`] — extract `_gckind` from a `Ptr.TO`
/// payload.  RPython exposes `_gckind` as an attribute on every
/// pointee (`Struct._gckind`, `Array._gckind`, etc.) so the
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

/// Per-function effect attributes — the metadata `call.py` /
/// `collectanalyze.py` read off `graph.func` (or, for a graph-less
/// external call, off `funcobj`). Carried directly on the
/// [`FunctionGraph`] (`graph.func`) for functions that have a graph, and
/// in `CallControl::external_funcobjs` keyed by `CallPath` for graph-less
/// externals (the `jit.*` intrinsics, `Vec::len`, etc.).
///
/// The raw source-attribute tokens (`_elidable_function_`,
/// `_jit_loop_invariant_`, the open `_jit_*_` policy hints like
/// `look_inside` / `unroll_safe` / `aroundstate`) also live on
/// [`FunctionGraph::hints`], the unbounded token bag that
/// [`crate::codewriter::policy`] matches against the synthesized
/// `SemanticFunction`. This struct is the typed carrier the CallControl
/// effect analyzers read instead of string-searching `hints` — matching
/// RPython's `getattr(func, <attr>)` reads — and is also the home for
/// the attributes that were previously kept in per-`CallControl`
/// GraphId-keyed side tables.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FuncEffects {
    /// `func.oopspec` (rlib/jit.py:250 `@oopspec(spec)`). Its presence is
    /// the builtin signal (call.py:135 `if hasattr(targetgraph.func,
    /// 'oopspec'): return 'builtin'`).
    pub oopspec: Option<String>,
    /// `argnames = ll_func.__code__.co_varnames[:nb_args]`
    /// (support.py:705) — declaration-order parameter names used by
    /// `parse_oopspec` to resolve identifier slots in the spec's `(...)`
    /// pattern. Empty when no `(...)`-bearing spec was registered.
    pub oopspec_argnames: Vec<String>,
    /// `func._gctransformer_hint_cannot_collect_` (collectanalyze.py:15)
    /// — `analyze_can_collect` returns False immediately.
    pub cannot_collect: bool,
    /// `funcobj.random_effects_on_gcobjs` (collectanalyze.py:21-25). Only
    /// meaningful for an external (graph-less) funcobj; a graph-bearing
    /// function derives can-collect by walking the graph instead.
    pub random_effects_on_gcobjs: bool,
    /// pyre `#[majit_macros::elidable_cannot_raise]` user assertion that
    /// the callee never raises (honoured by `getcalldescr`'s elidable
    /// branch before consulting `_canraise`).
    pub cannot_raise_assertion: bool,
    /// pyre `#[majit_macros::elidable_or_memerror]` user assertion that
    /// the callee raises only MemoryError.
    pub memerror_only_assertion: bool,
    /// `func._elidable_function_` (call.py:239) — the typed carrier the
    /// CallControl analyzers read (`is_elidable`, indirect-family check).
    /// The same `"elidable"` token also stays in [`FunctionGraph::hints`]
    /// for the policy `SemanticFunction` path.
    pub elidable: bool,
    /// `func._jit_loop_invariant_` (call.py:240).
    pub loop_invariant: bool,
    /// `func._gctransformer_hint_close_stack_` (call.py:129-134) — a
    /// close_stack callee must never produce a JitCode and is classified
    /// `Residual` by `guess_call_kind`.
    pub close_stack: bool,
}

impl FuncEffects {
    /// Fold `other`'s set attributes into `self`. Used when a graph is
    /// registered *after* a `mark_*` already recorded effects on the
    /// graph-less external funcobj record for the same path, so the typed
    /// carrier ends up registration-order-insensitive (every present
    /// value / set flag in `other` wins; cleared flags never unset `self`).
    pub fn merge_from(&mut self, other: &FuncEffects) {
        if other.oopspec.is_some() {
            self.oopspec = other.oopspec.clone();
        }
        if !other.oopspec_argnames.is_empty() {
            self.oopspec_argnames = other.oopspec_argnames.clone();
        }
        self.cannot_collect |= other.cannot_collect;
        self.random_effects_on_gcobjs |= other.random_effects_on_gcobjs;
        self.cannot_raise_assertion |= other.cannot_raise_assertion;
        self.memerror_only_assertion |= other.memerror_only_assertion;
        self.elidable |= other.elidable;
        self.loop_invariant |= other.loop_invariant;
        self.close_stack |= other.close_stack;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionGraph {
    pub name: String,
    /// Impl-block self-type root for graphs produced from `impl <T> { fn m(&self, ...) }`.
    /// Mirrors PyPy's `graph.func.im_class` access (the bound-method's class
    /// reference): RPython lifts `self` as `SomeInstance(getuniqueclassdef(im_class))`
    /// at `description.py:283-305 FunctionDesc.pycall`, then the rtyper resolves
    /// `self.<field>` against that ClassDef.  Pyre's per-graph
    /// `derive_subject_inputcells` (`flowspace_adapter.rs:1388`) uses this field
    /// to project the receiver inputarg as `SomeInstance(Some(classdef))` instead
    /// of the abstract `SomeInstance(None)` shell that `valuetype_to_someshell(Ref)`
    /// yields when no class hint is available.  `None` for free functions and
    /// synthetic test-fixture graphs.
    pub owner_root: Option<String>,
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
    /// The source-level return type string (Rust `syn::ReturnType`
    /// rendered through `qualified_full_type_string`). RPython
    /// equivalent: `funcptr._obj.TO.RESULT` on the
    /// `lltype.FuncType(ARGS, RESULT)` — every graph carries its result
    /// type intrinsically as `op.result.concretetype` of the
    /// returnblock's input variable.  Pyre stores the source string and
    /// projects to `Type` via `return_type_string_to_value_type` so the
    /// JIT codewriter's signature validator reads `FUNC.RESULT` directly
    /// off the callee graph.
    ///
    /// `None` for synthetic test-fixture graphs constructed via
    /// `FunctionGraph::new("name")`; production paths populate via
    /// `with_return_type(rt)` after construction (parse.rs + lib.rs
    /// free-function, trait-method, and inherent-method registration).
    pub return_type: Option<String>,
    /// Per-graph JIT hints — the `_jit_*_` / `_elidable_function_`
    /// attributes RPython `policy.py:48-62 look_inside_graph` reads off
    /// `graph.func`. Pyre carries them on the graph itself so
    /// [`crate::policy::JitPolicy::look_inside_graph`] reads the canonical
    /// carrier directly. Empty for graphs with no hints (the common case
    /// and all `FunctionGraph::new` fixtures).
    pub hints: Vec<String>,
    /// Per-function effect attributes RPython reads off `graph.func`
    /// (`func.oopspec`, `_gctransformer_hint_cannot_collect_`, …). Default
    /// (all unset) for `FunctionGraph::new` fixtures; production
    /// registration stamps these through `CallControl::mark_*`.
    pub func: FuncEffects,
}

impl FunctionGraph {
    pub fn new(name: impl Into<String>) -> Self {
        let entry = BlockId(0);
        let returnblock = BlockId(1);
        let exceptblock = BlockId(2);
        // Canonical inputargs (returnvar / etype / evalue) get named
        // Variables matching `flowspace/model.py:21-25`
        // (`Variable("returnvar")`, `Variable("etype")`,
        // `Variable("evalue")`).  Each one's `concretetype` cell
        // stays `None` until the rtyper resolves it; until then
        // `concretetype(v)` returns `ConcreteType::Unknown` (the
        // sentinel for the pre-`setconcretetype` window), which
        // matches the pre-rtyper Variable-with-no-`concretetype`
        // shape upstream.
        let var_returnvar = crate::flowspace::model::Variable::named("returnvar");
        let var_etype = crate::flowspace::model::Variable::named("etype");
        let var_evalue = crate::flowspace::model::Variable::named("evalue");
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
                    inputargs: vec![var_returnvar],
                    operations: Vec::new(),
                    exitswitch: None,
                    exits: Vec::new(),
                    framestate: None,
                    dead: false,
                },
                Block {
                    id: exceptblock,
                    inputargs: vec![var_etype, var_evalue],
                    operations: Vec::new(),
                    exitswitch: None,
                    exits: Vec::new(),
                    framestate: None,
                    dead: false,
                },
            ],
            notes: Vec::new(),
            return_type: None,
            owner_root: None,
            hints: Vec::new(),
            func: FuncEffects::default(),
        }
    }

    /// Builder-style setter for `return_type`. Used by production
    /// registration paths (parse.rs / lib.rs) where the source-level
    /// type is available; test fixtures that construct via
    /// `FunctionGraph::new("name")` may skip and leave `None`. The JIT
    /// codewriter signature validator leaves the result un-validated for
    /// `None`-carrying graphs.
    pub fn with_return_type(mut self, rt: impl Into<String>) -> Self {
        self.return_type = Some(rt.into());
        self
    }

    /// Builder-style setter for `hints`. Production registration paths
    /// (lib.rs free-function, trait-method, and inherent-method loops)
    /// stamp the parsed `_jit_*_` / `_elidable_function_` hints onto the
    /// graph; test fixtures may skip and leave the list empty.
    pub fn with_hints(mut self, hints: Vec<String>) -> Self {
        self.hints = hints;
        self
    }

    /// Builder-style setter for `owner_root` — the impl-block self-type
    /// root for graphs produced from `impl <T> { fn m(&self, ...) }`.
    /// `front::mir` sets this from the impl-method owner it records as
    /// `SemanticFunction.self_ty_root`;
    /// `derive_subject_inputcells` consults it to project the receiver
    /// inputarg as `SomeInstance(Some(getuniqueclassdef(im_class)))`.
    pub fn with_owner_root(mut self, owner: impl Into<String>) -> Self {
        self.owner_root = Some(owner.into());
        self
    }

    /// Return the canonical exception block and its `(etype, evalue)`
    /// inputarg Variables.
    ///
    /// RPython parity: `flowspace/model.py:21-25` `exceptblock` has
    /// two inputargs, `(etype, evalue)`, and exists eagerly on every
    /// graph.
    pub fn exceptblock_arg_vars(
        &self,
    ) -> (
        BlockId,
        crate::flowspace::model::Variable,
        crate::flowspace::model::Variable,
    ) {
        let inputargs = &self.block(self.exceptblock).inputargs;
        (self.exceptblock, inputargs[0].clone(), inputargs[1].clone())
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

    /// Create a merge block whose `inputargs` come from
    /// `framestate.getvariables()` — RPython `flowcontext.py:38
    /// SpamBlock(framestate)` parity:
    ///
    /// ```python
    /// class SpamBlock(Block):
    ///     def __init__(self, framestate):
    ///         Block.__init__(self, framestate.getvariables())
    ///         self.framestate = framestate
    /// ```
    ///
    /// The block's `inputargs` collect Variables from
    /// `locals + flatten(stack) + [exc_type, exc_value]` in positional
    /// order — the same `mergeable` shape `FrameState::getoutputargs`
    /// walks on the predecessor side.  Pairing the two halves of the
    /// recloseblock at this entry point keeps `link.args.len() ==
    /// target.inputargs.len()` (`simplify.py:513`) invariant for every
    /// predecessor of the new block.
    pub fn create_block_from_framestate(&mut self, fs: &FrameState) -> BlockId {
        let inputargs = fs.getvariables(self);
        let id = BlockId(self.blocks.len());
        self.blocks.push(Block {
            id,
            inputargs,
            operations: Vec::new(),
            exitswitch: None,
            exits: Vec::new(),
            framestate: Some(fs.clone()),
            dead: false,
        });
        id
    }

    /// Create a block with `num_args` fresh inputarg `Variable`s
    /// (Phi nodes), each minted via `alloc_value_var_with_type` with
    /// `ConcreteType::Unknown` and returned to the caller for direct
    /// use.  RPython parity: `flowmodel.py:130-145 Block(inputargs)`
    /// where each inputarg is a fresh Variable.
    pub fn create_block_with_arg_vars(
        &mut self,
        num_args: usize,
    ) -> (BlockId, Vec<crate::flowspace::model::Variable>) {
        let id = BlockId(self.blocks.len());
        let inputargs: Vec<crate::flowspace::model::Variable> = (0..num_args)
            .map(|_| self.alloc_value_var_with_type(ConcreteType::Unknown))
            .collect();
        self.blocks.push(Block {
            id,
            inputargs: inputargs.clone(),
            operations: Vec::new(),
            exitswitch: None,
            exits: Vec::new(),
            framestate: None,
            dead: false,
        });
        (id, inputargs)
    }

    /// Mint a fresh value [`crate::flowspace::model::Variable`] with
    /// `ConcreteType::Unknown`.
    pub fn alloc_value_var(&mut self) -> crate::flowspace::model::Variable {
        self.alloc_value_var_with_type(ConcreteType::Unknown)
    }

    /// Mint a fresh value [`crate::flowspace::model::Variable`] with its
    /// [`ConcreteType`] stamped at construction — pyre's analogue of
    /// upstream `Variable(concretetype=...)` (RPython
    /// `flowspace/model.py:Variable.__init__`).  The Variable's
    /// `concretetype` cell receives the canonical `LowLevelType`
    /// representative for `ty` (see [`concrete_to_canonical_lltype`]);
    /// the rtyper's `setconcretetype` later overrides it by writing the
    /// authoritative type onto this same cell.
    pub fn alloc_value_var_with_type(
        &mut self,
        ty: ConcreteType,
    ) -> crate::flowspace::model::Variable {
        let var = crate::flowspace::model::Variable::new();
        var.set_concretetype(concrete_to_canonical_lltype(ty));
        var
    }

    /// `Variable.concretetype` getter — reads the `Variable.concretetype`
    /// cell directly.  RPython's resolver iterates `Variable` instances
    /// directly (`rtyper.py:258 v.concretetype = ...`); use this when
    /// the caller already holds a `&Variable` (e.g.
    /// `OpKind::BinOp.lhs / .rhs`, `block.inputargs[i]`).
    pub fn concretetype_of(var: &crate::flowspace::model::Variable) -> ConcreteType {
        match var.concretetype.borrow().as_ref() {
            Some(lltype) => getkind(lltype),
            None => ConcreteType::Unknown,
        }
    }

    /// Variable-direct kind-publication helper — writes through the
    /// `Variable.concretetype` cell directly, preserving a richer
    /// rtyper-written `LowLevelType` when its `getkind` already
    /// matches `ty`.  Used by passes that already hold a `&Variable`
    /// and want to skip the slot → Variable lookup.
    pub fn set_concretetype_of_inline(var: &crate::flowspace::model::Variable, ty: ConcreteType) {
        let preserve_existing = matches!(
            (ty, var.concretetype.borrow().as_ref()),
            (kind, Some(existing)) if getkind(existing) == kind
        );
        if !preserve_existing {
            let synthetic = concrete_to_canonical_lltype(ty);
            var.set_concretetype(synthetic);
        }
    }

    /// Walk the startblock-reachable block closure (`iterblocks()`
    /// parity, `rpython/flowspace/model.py:66`) and collect every distinct
    /// [`crate::flowspace::model::Variable`] it references: block
    /// `inputargs`, operation operands
    /// ([`crate::inline::op_variable_refs`]) and results, link `args` /
    /// exception extravars, and the `exitswitch` condition.  Distinct by
    /// identity, in first-appearance DFS order.
    ///
    /// The canonical value enumeration for passes that walk the graph's
    /// values (regalloc coloring, `Variable.annotation` reset /
    /// commit).  Reads the value set straight off the IR.  Values
    /// minted but never referenced are naturally absent — every
    /// consumer filters such values out anyway (`getcolor` returns
    /// `None`, `annotation` is empty).
    pub fn iter_variables(&self) -> Vec<crate::flowspace::model::Variable> {
        use crate::flowspace::model::Variable;
        let mut seen: std::collections::HashSet<Variable> = std::collections::HashSet::new();
        let mut out: Vec<Variable> = Vec::new();
        let push = |v: Variable,
                    seen: &mut std::collections::HashSet<Variable>,
                    out: &mut Vec<Variable>| {
            if seen.insert(v.clone()) {
                out.push(v);
            }
        };
        // `iterblocks()` parity (`rpython/flowspace/model.py:66`): walk only
        // the startblock-reachable closure over `Block.exits`, id-keyed
        // because block ids need not be index-aligned
        // (`flowspace_adapter::reachable_block_ids`).  Unreachable lowered
        // blocks (orphan `on_unwind` cleanup, pruned `SwitchInt` arms) carry
        // no annotated value — `remove_dead_blocks` drops them before rtyping.
        let by_id: std::collections::HashMap<BlockId, &Block> =
            self.blocks.iter().map(|b| (b.id, b)).collect();
        let mut block_seen: std::collections::HashSet<BlockId> = std::collections::HashSet::new();
        let mut stack = vec![self.startblock];
        while let Some(bid) = stack.pop() {
            if !block_seen.insert(bid) {
                continue;
            }
            let Some(block) = by_id.get(&bid) else {
                continue;
            };
            for v in block.input_variables() {
                push(v.clone(), &mut seen, &mut out);
            }
            for op in &block.operations {
                for v in crate::inline::op_variable_refs(&op.kind) {
                    push(v, &mut seen, &mut out);
                }
                if let Some(result) = &op.result {
                    push(result.clone(), &mut seen, &mut out);
                }
            }
            for link in &block.exits {
                for arg in &link.args {
                    if let Some(v) = arg.as_variable() {
                        push(v.clone(), &mut seen, &mut out);
                    }
                }
                if let Some(v) = link.last_exception.as_ref().and_then(LinkArg::as_variable) {
                    push(v.clone(), &mut seen, &mut out);
                }
                if let Some(v) = link.last_exc_value.as_ref().and_then(LinkArg::as_variable) {
                    push(v.clone(), &mut seen, &mut out);
                }
            }
            if let Some(ExitSwitch::Value(cond)) = &block.exitswitch {
                push(cond.clone(), &mut seen, &mut out);
            }
            stack.extend(block.exits.iter().rev().map(|e| e.target));
        }
        out
    }

    /// Walk the startblock-reachable block closure in `iterblocks()`
    /// order (`rpython/flowspace/model.py:66`) and return the visited
    /// [`BlockId`]s.  The startblock is yielded first, then each block's
    /// exits are pushed reversed so the first exit is visited first —
    /// the canonical DFS order RPython passes consume.  Block ids need
    /// not be index-aligned with `self.blocks` storage order, so this
    /// is keyed by id, not by Vec position.
    pub fn iterblocks_order(&self) -> Vec<BlockId> {
        let by_id: std::collections::HashMap<BlockId, &Block> =
            self.blocks.iter().map(|b| (b.id, b)).collect();
        let mut seen: std::collections::HashSet<BlockId> = std::collections::HashSet::new();
        let mut order: Vec<BlockId> = Vec::new();
        let mut stack = vec![self.startblock];
        while let Some(bid) = stack.pop() {
            if !seen.insert(bid) {
                continue;
            }
            order.push(bid);
            if let Some(block) = by_id.get(&bid) {
                stack.extend(block.exits.iter().rev().map(|e| e.target));
            }
        }
        order
    }

    /// Push an op whose fresh result `Variable` is minted in place
    /// when `has_result` is true; callers receive that `Variable`
    /// directly.
    pub fn push_op_var(
        &mut self,
        block: BlockId,
        kind: OpKind,
        has_result: bool,
    ) -> Option<crate::flowspace::model::Variable> {
        let result_var = has_result.then(|| self.alloc_value_var());
        self.blocks[block.0].operations.push(SpaceOperation {
            result: result_var.clone(),
            kind,
        });
        result_var
    }

    /// Push an op with a caller-supplied `result_var`.  Used when the
    /// `Variable` was allocated upstream (e.g. `FrameState::union`
    /// pre-allocates phi variables) and the op now needs to be emitted
    /// with the same `Variable` identity.
    pub fn push_op_with_result_var(
        &mut self,
        block: BlockId,
        kind: OpKind,
        result_var: crate::flowspace::model::Variable,
    ) {
        self.blocks[block.0].operations.push(SpaceOperation {
            result: Some(result_var),
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
    pub fn set_goto(
        &mut self,
        block: BlockId,
        target: BlockId,
        args: Vec<crate::flowspace::model::Variable>,
    ) {
        // `flowspace/model.py:114 Link.__init__` asserts
        // `len(args) == len(target.inputargs)` at construction time.
        // `set_goto` is a production-only wrapper whose `target` is
        // a fully-formed block (inputargs already populated by the
        // caller path); enforce the assert here so a mismatch
        // surfaces at the recloseblock site rather than as an
        // unbalanced subst dict in `eliminate_empty_blocks`
        // (`simplify.py:513`).  The inliner uses
        // `set_control_flow_metadata` directly when threading
        // remapped exits whose targets may transiently lack
        // inputargs; that lower-level path intentionally bypasses
        // this assert.
        let target_inputarg_count = self.block(target).inputargs.len();
        assert_eq!(
            args.len(),
            target_inputarg_count,
            "set_goto: args.len() ({}) != target.inputargs.len() ({}) — \
             block {:?} → target {:?} on graph {:?}",
            args.len(),
            target_inputarg_count,
            block,
            target,
            self.name,
        );
        let link = Link::from_variables(self, args, target, None);
        self.set_control_flow_metadata(block, None, vec![link]);
    }

    /// Close `block` with a single-exit Link into `target_block` whose
    /// `args` are derived from `pred_state.getoutputargs(target_state,
    /// self)`.  Direct port of `flowcontext.py:438`:
    ///
    /// ```python
    /// outputargs = currentstate.getoutputargs(newstate)
    /// block.recloseblock(Link(outputargs, newblock))
    /// ```
    ///
    /// `pred_state` is the predecessor's exit framestate (upstream
    /// `currentstate`); `target_state` is the merge block's entry
    /// framestate (upstream `newstate`).  Variable cells route into
    /// `LinkArg::Value(Variable)` (locals via `LinkArg::value(graph,
    /// vid)` lookup, stack / exception via `hlvalue_to_linkarg`);
    /// Constant cells in the stack / exception projections flow
    /// through `LinkArg::Const` directly.  Mixed-shape outputargs are
    /// threaded into a single Link via `Link::new_mixed`, mirroring
    /// upstream's `Link(outputargs, newblock)` constructor where
    /// `outputargs: List[Hlvalue]` carries Variable + Constant cells
    /// side by side.  Use this in place of `set_goto(.., args)` at
    /// every merge-close site that has both predecessor and target
    /// framestates in hand.
    pub fn set_goto_from_framestate(
        &mut self,
        block: BlockId,
        target_block: BlockId,
        pred_state: &FrameState,
        target_state: &FrameState,
    ) {
        let outputargs = pred_state.getoutputargs(target_state, self);
        // Backfill ancestor-defined Variable args into `block`'s
        // inputargs + predecessor link args via the eager-threading
        // helper.  When `target_state.mergeable` carries a carry-through
        // Variable (e.g. the function parameter `cond` that flows
        // unchanged through both arms of an `if`), `getoutputargs`
        // returns it as `LinkArg::Value(cond)` — but `block` may not
        // define `cond` locally, so the Link.args would reference an
        // undefined operand (`flowspace_adapter.rs:1324`).  Walking
        // back through the predecessor chain to add `cond` as a
        // `block.inputargs` entry restores the
        // `flowspace/flowcontext.py:407-408 setstate(block.framestate)`
        // shape that the AST walker doesn't produce eagerly (task #91
        // prereq).
        for arg in &outputargs {
            if let LinkArg::Value(var) = arg {
                self.ensure_variable_at_block(block, var);
            }
        }
        // `flowspace/model.py:114 Link.__init__` asserts
        // `len(args) == len(target.inputargs)` at construction time.
        // Run the same check here so a framestate-driven recloseblock
        // catches the mismatch at the merge-close site rather than
        // surfacing it downstream as an unbalanced subst dict in
        // `eliminate_empty_blocks` (`simplify.py:513`).
        let target_inputarg_count = self.block(target_block).inputargs.len();
        assert_eq!(
            outputargs.len(),
            target_inputarg_count,
            "set_goto_from_framestate: outputargs.len() ({}) != target.inputargs.len() ({}) — \
             block {:?} → target {:?} on graph {:?}",
            outputargs.len(),
            target_inputarg_count,
            block,
            target_block,
            self.name,
        );
        let link = Link::new_mixed(outputargs, target_block, None);
        self.set_control_flow_metadata(block, None, vec![link]);
    }

    /// `flowcontext.py:440 currentblock.closeblock(Link(outputargs, block))`
    /// — compute link args via `getoutputargs` and install the Link
    /// WITHOUT calling [`Self::ensure_variable_at_block`].
    ///
    /// RPython's `closeblock` (`model.py:246`) asserts the block is open
    /// (exits empty) before installing exits.  Use [`Self::recloseblock`]
    /// or [`Self::recloseblock_link`] when re-closing an already-closed
    /// block.
    pub fn closeblock_link(
        &mut self,
        block: BlockId,
        target_block: BlockId,
        pred_state: &FrameState,
        target_state: &FrameState,
    ) {
        let outputargs = pred_state.getoutputargs(target_state, self);
        let target_inputarg_count = self.block(target_block).inputargs.len();
        assert_eq!(
            outputargs.len(),
            target_inputarg_count,
            "closeblock_link: outputargs.len() ({}) != target.inputargs.len() ({}) — \
             block {:?} → target {:?} on graph {:?}",
            outputargs.len(),
            target_inputarg_count,
            block,
            target_block,
            self.name,
        );
        let link = Link::new_mixed(outputargs, target_block, None);
        self.closeblock(block, vec![link]);
    }

    /// Like [`Self::closeblock_link`] but for blocks that may already be
    /// closed — delegates to [`Self::recloseblock`] instead of
    /// [`Self::closeblock`].  `model.py:250 recloseblock`.
    pub fn recloseblock_link(
        &mut self,
        block: BlockId,
        target_block: BlockId,
        pred_state: &FrameState,
        target_state: &FrameState,
    ) {
        let outputargs = pred_state.getoutputargs(target_state, self);
        let target_inputarg_count = self.block(target_block).inputargs.len();
        assert_eq!(
            outputargs.len(),
            target_inputarg_count,
            "recloseblock_link: outputargs.len() ({}) != target.inputargs.len() ({}) — \
             block {:?} → target {:?} on graph {:?}",
            outputargs.len(),
            target_inputarg_count,
            block,
            target_block,
            self.name,
        );
        let link = Link::new_mixed(outputargs, target_block, None);
        self.recloseblock(block, vec![link]);
    }

    /// Backfill `var` as a definition reachable at `block`.
    ///
    /// Cycle-safe top-down: adds `var` to `block.inputargs` BEFORE
    /// recursing into predecessors so a back-edge that reaches `block`
    /// again finds the in-progress slot via
    /// [`Self::variable_defined_in_block`] and short-circuits.  For
    /// each predecessor edge `(pred_block, exit_idx)`, recursively
    /// ensures `pred_block` defines `var`, then appends
    /// `LinkArg::Value(var)` to `pred_block.exits[exit_idx].args` so
    /// the arity invariant `len(link.args) == len(target.inputargs)`
    /// is preserved.
    ///
    /// Returns `true` when `var` is now reachable at `block` (was
    /// already defined, or newly threaded).  Returns `false` only when
    /// `block` has no predecessors AND `var` is not defined locally
    /// — a malformed graph state callers should treat as a bug.
    ///
    /// `flowspace/flowcontext.py:407-408 setstate(block.framestate)`
    /// populates `block.inputargs` with all carry-through Variables at
    /// block entry by walking top-down.  Pyre's AST walker doesn't do
    /// this eagerly (task #91), so when a framestate-driven merge
    /// (`create_block_from_framestate`) surfaces an ancestor-defined
    /// Variable in `block.inputargs`, this helper backfills the
    /// predecessor chain bottom-up to the same final graph shape.
    pub fn ensure_variable_at_block(
        &mut self,
        block: BlockId,
        var: &crate::flowspace::model::Variable,
    ) -> bool {
        if self.variable_defined_in_block(block, var) {
            return true;
        }
        let pred_edges: Vec<(BlockId, usize)> = self
            .blocks
            .iter()
            .flat_map(|b| {
                let bid = b.id;
                b.exits.iter().enumerate().filter_map(move |(i, exit)| {
                    if exit.target == block {
                        Some((bid, i))
                    } else {
                        None
                    }
                })
            })
            .collect();
        if pred_edges.is_empty() {
            return false;
        }
        // If `block` is a phi-block (has any `OpKind::Input` op), every
        // inputarg must be paired with an `OpKind::Input { name, .. }` op.
        // When this helper adds a fresh inputarg via the recursive backfill
        // we must also emit the paired op so the invariant holds; otherwise
        // `closeblock_link` / `getoutputargs` panics on arity mismatch.
        let is_phi_block = self
            .block(block)
            .operations
            .iter()
            .any(|op| matches!(op.kind, OpKind::Input { .. }));
        self.block_mut(block).inputargs.push(var.clone());
        if is_phi_block {
            let name = self.value_name_for(var).unwrap_or_default();
            self.push_op_with_result_var(
                block,
                OpKind::Input {
                    name,
                    ty: ValueType::Unknown,
                    class_root: None,
                },
                var.clone(),
            );
        }
        for (pred_block, exit_idx) in pred_edges {
            let ok = self.ensure_variable_at_block(pred_block, var);
            if !ok {
                // The Variable's source name is read identity-first off
                // its `OpKind::Input` op via `value_name_for`; the panic
                // identifies the variable by `id` + source name.
                let name = self.value_name_for(var);
                // Dump the blocks whose framestate references `var` and the
                // blocks that actually define it.  Surfaces stale carry-
                // through bindings (e.g. a sibling control-flow path's
                // local leaking into a slot whose backing Variable is no
                // longer reachable via the predecessor chain) without
                // forcing the engineer to rebuild a custom dump.
                let mut framestate_blocks: Vec<BlockId> = Vec::new();
                for b in &self.blocks {
                    let Some(fs) = b.framestate.as_ref() else {
                        continue;
                    };
                    let in_locals = fs.entries.iter().any(|e| e.as_ref() == Some(var));
                    let in_stack = crate::flowspace::framestate::recursively_flatten(&fs.stack)
                        .iter()
                        .any(|h| {
                            matches!(h, crate::flowspace::model::Hlvalue::Variable(v) if v == var)
                        });
                    if in_locals || in_stack {
                        framestate_blocks.push(b.id);
                    }
                }
                let mut def_blocks: Vec<BlockId> = Vec::new();
                for b in &self.blocks {
                    if b.inputargs.contains(var)
                        || b.operations
                            .iter()
                            .any(|op| op.result.as_ref() == Some(var))
                    {
                        def_blocks.push(b.id);
                    }
                }
                panic!(
                    "ensure_variable_at_block: predecessor {:?} cannot define \
                     Variable id={} (name={:?}) when threading to block {:?} on graph {:?} \
                     — graph corruption: no transitive predecessor chain leads \
                     to a definition site. \
                     Variable appears in framestate of blocks {:?} and in inputargs/operations of blocks {:?}.",
                    pred_block,
                    var.id(),
                    name,
                    block,
                    self.name,
                    framestate_blocks,
                    def_blocks,
                );
            }
            self.block_mut(pred_block).exits[exit_idx]
                .args
                .push(LinkArg::Value(var.clone()));
        }
        true
    }

    /// `true` iff `var` is defined inside `block` — either as a
    /// `Block.inputargs` entry or as the result of one of its
    /// `operations`.  `flowspace/model.py:checkgraph` —
    /// every operand referenced from a block must be defined in that
    /// block.
    pub fn variable_defined_in_block(
        &self,
        block: BlockId,
        var: &crate::flowspace::model::Variable,
    ) -> bool {
        let b = self.block(block);
        if b.inputargs.contains(var) {
            return true;
        }
        b.operations
            .iter()
            .any(|op| op.result.as_ref() == Some(var))
    }

    /// Shorthand for the boolean-branch shape — two Links with
    /// `Bool(false)` / `Bool(true)` exitcases, `exitswitch =
    /// ExitSwitch::Value(cond)`.  Upstream equivalent:
    /// `block.exitswitch = cond;
    ///  block.closeblock(Link(false_args, if_false, False),
    ///                   Link(true_args,  if_true,  True))`
    /// (`flowspace/model.py:175-180` + `:304`).
    ///
    /// RPython `flowcontext.py:744-779` unconditionally evaluates
    /// `op.bool(w_value).eval(self)` before `guessbool`, so every
    /// `block.exitswitch` Variable is the result of a `bool` HighLevelOp.
    /// The rtyper then specialises `bool` per repr (`rmodel.py:251-260
    /// CanBeNull → ptr_nonzero`; `rint.py IntegerRepr → identity`;
    /// `rstr.py → str_nonzero`; etc.).  Pyre's flatten consumer asserts
    /// `block.exitswitch.concretetype == lltype.Bool`
    /// (`flatten.py:248`) — without the unconditional `bool` wrap, a
    /// composite-pattern `match` / `if let` scrutinee of `Ref` kind
    /// reaches `FlatOp::GotoIfNot` (`assembler.rs:559-578`), the
    /// hard-coded `goto_if_not/iL` opname forces the walker into the
    /// wrong register bank, and every LoadAttr/StoreAttr arm aborts with
    /// `RegisterOutOfRange` (issue #115).  Routing the cond through a
    /// `bool` hop here re-establishes the upstream invariant in one
    /// place; the rtyper handles the per-repr specialisation downstream.
    pub fn set_branch(
        &mut self,
        block: BlockId,
        cond: crate::flowspace::model::Variable,
        if_true: BlockId,
        true_args: Vec<crate::flowspace::model::Variable>,
        if_false: BlockId,
        false_args: Vec<crate::flowspace::model::Variable>,
    ) {
        // upstream: `op.bool(w_value).eval(self)` — append `bool` hop to
        // `block.operations` before installing it as exitswitch.  The
        // wrap is idempotent (a `bool(bool(_))` chain folds to identity
        // through `IntegerRepr::rtype_bool` / `BoolRepr::rtype_bool`),
        // mirroring upstream which also wraps unconditionally.
        let cond = self
            .push_op_var(
                block,
                OpKind::UnaryOp {
                    op: "bool".into(),
                    operand: cond,
                    result_ty: ValueType::Bool,
                },
                true,
            )
            .expect("UnaryOp { op: \"bool\", .. } produces a value");
        // `flowspace/model.py:114 Link.__init__` arity assert per
        // arm — same rationale as `set_goto`.
        let true_inputarg_count = self.block(if_true).inputargs.len();
        assert_eq!(
            true_args.len(),
            true_inputarg_count,
            "set_branch: true_args.len() ({}) != if_true.inputargs.len() ({}) — \
             block {:?} → if_true {:?} on graph {:?}",
            true_args.len(),
            true_inputarg_count,
            block,
            if_true,
            self.name,
        );
        let false_inputarg_count = self.block(if_false).inputargs.len();
        assert_eq!(
            false_args.len(),
            false_inputarg_count,
            "set_branch: false_args.len() ({}) != if_false.inputargs.len() ({}) — \
             block {:?} → if_false {:?} on graph {:?}",
            false_args.len(),
            false_inputarg_count,
            block,
            if_false,
            self.name,
        );
        let false_link =
            Link::from_variables(self, false_args, if_false, Some(ExitCase::Bool(false)))
                .with_llexitcase_from_exitcase();
        let true_link = Link::from_variables(self, true_args, if_true, Some(ExitCase::Bool(true)))
            .with_llexitcase_from_exitcase();
        self.set_control_flow_metadata(
            block,
            Some(ExitSwitch::Value(cond)),
            vec![false_link, true_link],
        );
    }

    /// Route a return through the graph's canonical `returnblock`.
    ///
    /// RPython `flowcontext.py:687-689 RETURN_VALUE` pops `w_returnvalue`
    /// from the stack and wraps it in `Return(w_returnvalue)`; in
    /// `flowcontext.py:1232-1236 Return.nomoreblocks`, the link is built
    /// as `Link([w_result], ctx.graph.returnblock)`.  For `def foo():
    /// pass`, the bytecode is `LOAD_CONST None; RETURN_VALUE`, so
    /// `w_result` is `Constant(None)` and `Link.args = [Constant(None)]`
    /// — an `Hlvalue::Constant`, not a freshly allocated Variable.
    ///
    /// `value == None` wires `LinkArg::Const(ConstValue::None)` with
    /// `concretetype = Void` into `Link.args[0]` directly, matching
    /// upstream's `Constant(None)`-on-the-stack shape with the
    /// `NoneRepr.lowleveltype = Void` (`rpython/rtyper/rnone.py`)
    /// kind stamped at construction.  Pyre's downstream consumers
    /// (`flatten::constant_kind`, assembler `emit_const`) honour the
    /// explicit concretetype so the constant flows through
    /// `FlatOp::VoidReturn` instead of the `constvalue_kind(None) → 'r'`
    /// fallback that would otherwise route a Ref-kind void constant
    /// into the unsupported `emit_const_r(None)` pool.
    pub fn set_return(&mut self, block: BlockId, value: Option<crate::flowspace::model::Variable>) {
        let returnblock = self.returnblock;
        let arg = match value {
            Some(var) => LinkArg::Value(var),
            None => LinkArg::Const(crate::flowspace::model::Constant::with_concretetype(
                ConstValue::None,
                crate::translator::rtyper::lltypesystem::lltype::VOID,
            )),
        };
        let link = Link::new_mixed(vec![arg], returnblock, None);
        self.set_control_flow_metadata(block, None, vec![link]);
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
    /// Emits an unconditional Link to `graph.exceptblock` carrying the
    /// `RaiseImplicit` `(w_type, w_value)` pair: the `AssertionError`
    /// *class* Constant and a separate `AssertionError(msg)` *instance*
    /// Constant (`flowcontext.py:1280-1282`).  The two slots are never
    /// the same value — `w_value` is an exception instance, not the
    /// class.  A bare raise with no concrete payload — a Rust
    /// `UnwindResume` / `Abort` — has no source exception class, so
    /// `exc_cls` is the upstream `else: Exception` default and the
    /// instance message is the fixed `"implicit Exception shouldn't
    /// occur"`.  Constants are always resolvable by the cutover adapter
    /// (`flowspace_adapter.rs:1686 LinkArg::Const`), unlike the fresh
    /// `alloc_value_var()` placeholders this previously emitted: those
    /// had no producing operation and tripped the adapter's "undefined
    /// operand as Link.args" invariant on every reachable raise block.
    /// `last_exception` / `last_exc_value` extravars are not usable here:
    /// `checkgraph` (`flowspace/model.rs:4693-4703`) permits them only
    /// on the exception exits of a `canraise` block, not on this
    /// unconditional single exit.
    ///
    /// The `_reason` string is retained for optional GraphTransformNote
    /// annotations (see `jtransform.rs::transform_graph`'s abort note);
    /// pass `""` when not applicable.
    ///
    /// Used only where no concrete exception payload is available at
    /// the raise site — the evaluated arguments path should call
    /// `set_raise_values` instead.
    pub fn set_raise(&mut self, block: BlockId, _reason: &str) {
        use crate::flowspace::model::{HOST_ENV, HostObject};
        let exceptblock = self.exceptblock;
        // w_type = Constant(AssertionError), w_value = Constant(AssertionError(msg)).
        let etype = LinkArg::from(ConstValue::builtin("AssertionError"));
        let cls = HOST_ENV
            .lookup_builtin("AssertionError")
            .unwrap_or_else(|| panic!("HOST_ENV missing builtin AssertionError"));
        let message = ConstValue::byte_str("implicit Exception shouldn't occur");
        let instance = HostObject::new_instance(cls, vec![message]);
        let evalue = LinkArg::from(ConstValue::HostObject(instance));
        let link = Link::new_mixed(vec![etype, evalue], exceptblock, None);
        self.set_control_flow_metadata(block, None, vec![link]);
    }

    /// Terminate `block` with the implicit-exception raise shape of
    /// RPython `RaiseImplicit.nomoreblocks`
    /// (`flowspace/flowcontext.py:1271-1284`):
    ///
    /// ```python
    /// msg = "implicit %s shouldn't occur" % exc_cls.__name__
    /// w_type = Constant(AssertionError)
    /// w_value = Constant(AssertionError(msg))
    /// link = Link([w_type, w_value], ctx.graph.exceptblock)
    /// ctx.recorder.crnt_block.closeblock(link)
    /// ```
    ///
    /// Both link args are Constants with `args[0]` carrying the
    /// `AssertionError` class itself — the shape
    /// [`remove_assertion_errors`] matches to prune the branch under
    /// the "AssertionError shouldn't occur at run-time" assumption
    /// (`simplify.py:321-346`).
    pub fn set_raise_implicit(&mut self, block: BlockId, exc_name: &str) {
        use crate::flowspace::model::{Constant, HOST_ENV, HostObject};
        let assert_err = HOST_ENV
            .lookup_builtin("AssertionError")
            .expect("HOST_ENV missing AssertionError");
        let msg = format!("implicit {exc_name} shouldn't occur");
        let w_type = LinkArg::Const(Constant::new(ConstValue::HostObject(assert_err.clone())));
        let w_value = LinkArg::Const(Constant::new(ConstValue::HostObject(
            HostObject::new_instance(assert_err, vec![ConstValue::UniStr(msg)]),
        )));
        let link = Link::new_mixed(vec![w_type, w_value], self.exceptblock, None);
        self.set_control_flow_metadata(block, None, vec![link]);
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
    /// Variables into the exceptblock Link so the exception payload
    /// is preserved, not discarded in favor of fresh `alloc_value_var()`
    /// placeholders as `set_raise` does.
    pub fn set_raise_values(
        &mut self,
        block: BlockId,
        etype: crate::flowspace::model::Variable,
        evalue: crate::flowspace::model::Variable,
    ) {
        self.set_goto(block, self.exceptblock, vec![etype, evalue]);
    }

    pub fn block(&self, block: BlockId) -> &Block {
        &self.blocks[block.0]
    }

    /// `Variable.rename(name)` (model.py:311-326) — name the Variable
    /// object's `_name`. Identity-shared `_name` (`Rc<RefCell<..>>`)
    /// makes the rename visible through every clone, including the
    /// `Block.inputargs` entry and the registry clone, so the cleaned
    /// name reaches `backendopt/ssa`. `rename` is first-wins ("don't
    /// rename several times"), so re-naming an already-named Variable
    /// is a no-op.
    pub fn name_value_var(
        &mut self,
        var: &crate::flowspace::model::Variable,
        name: impl Into<String>,
    ) {
        let mut v = var.clone();
        v.rename(&name.into());
    }

    /// Source name of `var` if it entered SSA through an
    /// `OpKind::Input` op, resolved by Variable identity directly — no
    /// graph-slot round trip. This is the raw, uncleaned name (the
    /// equivalent of `co_varnames`, read by `signature_for` and the
    /// flowspace adapter's same-name dedup); it is distinct from the
    /// cleaned `Variable._name` carried on the object via `rename`
    /// (read by `backendopt/ssa`). Values that do not enter through an
    /// `Input` op (e.g. a `let` binding to an arithmetic result) have
    /// no source name.
    ///
    /// The `Input` op is the *only* carrier of the raw name: `rename`
    /// stores `clean_name(name)` on `Variable._name` (which appends a
    /// trailing `_` and maps non-identifier chars to `_`), so the
    /// object name cannot reconstruct `co_varnames`. For startblock
    /// formal parameters the recovery is exact 1:1 — the param-binding
    /// loop (`lower_expr_into_graph_with_signature` / `build_function_
    /// graph`) emits a named `Input` op for every inputarg — so the
    /// `arg{N}` fallback in `signature_for_graph` is unreachable for
    /// well-formed graphs. `pygraph.py:16` instead names the initial-
    /// block locals straight from `code.co_varnames` and stores
    /// `code.signature` on the `PyGraph` wrapper; pyre's lifted callee
    /// graphs carry no such wrapper, so the raw name is recovered from
    /// the `Input` op until a `PyGraph`-equivalent signature store
    /// lands. Orthodox reader for callers that already hold the
    /// `Variable` (e.g. `signature_for_graph` walking
    /// `startblock.inputargs`); RPython reads the source name off the
    /// Variable / `co_varnames`, never via an integer value index.
    pub fn value_name_for(&self, var: &crate::flowspace::model::Variable) -> Option<String> {
        // Unnamed values (arithmetic temporaries, constants) carry no
        // source name; `Variable.renamed` is the O(1) gate that keeps
        // the `Input`-op lookup off the hot path for the common case.
        if !var.renamed() {
            return None;
        }
        let var_id = var.id();
        for block in &self.blocks {
            for op in &block.operations {
                if let OpKind::Input { name, .. } = &op.kind {
                    if op.result.as_ref().map(|r| r.id()) == Some(var_id) {
                        return Some(name.clone());
                    }
                }
            }
        }
        None
    }

    /// `Variable.rename(name)` — alias of [`Self::name_value_var`].
    /// Both honour the first-wins idempotency of `rename`; used by
    /// `FrameState::copy` (name-prefix carry-over) and the `mergeblock`
    /// generalize arm (`flowcontext.py:444-447`).
    pub fn rename_value_var(
        &mut self,
        var: &crate::flowspace::model::Variable,
        name: impl Into<String>,
    ) {
        self.name_value_var(var, name);
    }

    pub fn block_mut(&mut self, block: BlockId) -> &mut Block {
        &mut self.blocks[block.0]
    }

    /// Append `var` to `block.inputargs` — callers that synthesise
    /// inputargs at front-end / merge time pass the freshly minted
    /// Variable directly onto the upstream-orthodox `Vec<Variable>`
    /// storage.
    pub fn push_inputarg_var(&mut self, block: BlockId, var: crate::flowspace::model::Variable) {
        self.block_mut(block).inputargs.push(var);
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
            let args: Vec<String> = block.inputargs.iter().map(|v| v.name()).collect();
            if args.is_empty() {
                out.push_str(&format!("  Block {}:\n", block.id.0));
            } else {
                out.push_str(&format!("  Block {}({}):\n", block.id.0, args.join(", ")));
            }
            for op in &block.operations {
                let result = op
                    .result
                    .as_ref()
                    .map(|v| format!("{} = ", v.name()))
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `CallTarget::SyntheticTransparentCtor::path_segments()` must
    /// return the full `(owner_path..., name)` join — `front/mir.rs`
    /// `Aggregate` / `ShallowInitBox` lowerings rely on the qualified
    /// identity to distinguish same-leaf ctors across owner enums
    /// (`StepResult::Continue` vs `JitAction::Continue`).  Collapsing
    /// to the bare leaf would collide their `HostObject` /
    /// `getdesc(pyobj)` keys (`bookkeeper.py:353`).
    #[test]
    fn synthetic_transparent_ctor_path_segments_preserve_owner_path() {
        let bare = CallTarget::synthetic_transparent_ctor("Continue");
        assert_eq!(bare.path_segments(), Some(vec!["Continue"]));

        let owned = CallTarget::synthetic_transparent_ctor_with_owner(
            vec!["StepResult".to_string()],
            "Continue",
        );
        assert_eq!(owned.path_segments(), Some(vec!["StepResult", "Continue"]));

        let nested = CallTarget::synthetic_transparent_ctor_with_owner(
            vec!["outer".to_string(), "Inner".to_string()],
            "Leaf",
        );
        assert_eq!(nested.path_segments(), Some(vec!["outer", "Inner", "Leaf"]));
    }

    #[test]
    fn graph_allocates_values_and_blocks() {
        let mut graph = FunctionGraph::new("demo");
        let entry = graph.startblock;
        let cond_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "x".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let next = graph.create_block();
        graph.set_branch(entry, cond_var, next, vec![], next, vec![]);
        assert_eq!(graph.blocks.len(), 4);
        // `set_branch` mirrors RPython `flowcontext.py:744-779`
        // `op.bool(w_value).eval(self)` — every exitswitch is the result
        // of an appended `bool` HighLevelOp, so a branching block carries
        // the original Input op plus the bool wrap (2 ops total).
        assert_eq!(graph.block(entry).operations.len(), 2);
        assert_eq!(graph.block(graph.returnblock).inputargs.len(), 1);
        assert_eq!(graph.block(graph.exceptblock).inputargs.len(), 2);
    }

    #[test]
    fn set_control_flow_metadata_stamps_prevblock() {
        let mut graph = FunctionGraph::new("demo");
        let entry = graph.startblock;
        let next = graph.create_block();
        let link = Link::from_variables(&graph, vec![], next, None);
        graph.set_control_flow_metadata(entry, None, vec![link]);
        assert_eq!(graph.block(entry).exits[0].prevblock, Some(entry));
    }

    #[test]
    #[should_panic(expected = "output args mismatch")]
    fn link_new_panics_on_output_arg_mismatch() {
        let mut graph = FunctionGraph::new("demo");
        let (target, _) = graph.create_block_with_arg_vars(1);
        let _ = Link::from_variables(&graph, vec![], target, None);
    }

    #[test]
    fn set_return_routes_non_void_returns_via_returnblock() {
        let mut graph = FunctionGraph::new("demo");
        let entry = graph.startblock;
        let value_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "x".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(value_var.clone()));
        // Upstream `flowspace/model.py:171-180` identifies the routed
        // return by Block.exits carrying a single Link(value, returnblock)
        // with exitswitch=None.
        let entry_block = graph.block(entry);
        assert!(entry_block.exitswitch.is_none());
        assert_eq!(entry_block.exits.len(), 1);
        assert_eq!(entry_block.exits[0].prevblock, Some(entry));
        assert_eq!(entry_block.exits[0].target, graph.returnblock);
        assert_eq!(entry_block.exits[0].args, vec![LinkArg::Value(value_var)]);
    }

    #[test]
    fn set_raise_routes_assertionerror_class_and_instance_to_exceptblock() {
        use crate::flowspace::model::ConstValue;
        let mut graph = FunctionGraph::new("demo");
        let entry = graph.startblock;
        graph.set_raise(entry, "");
        // RaiseImplicit.nomoreblocks (flowcontext.py:1280-1282): the
        // exceptblock Link carries the AssertionError *class* as w_type
        // and a distinct AssertionError *instance* as w_value — never the
        // class in both slots.
        let entry_block = graph.block(entry);
        assert_eq!(entry_block.exits.len(), 1);
        let link = &entry_block.exits[0];
        assert_eq!(link.target, graph.exceptblock);
        assert_eq!(link.args.len(), 2);
        let host_of = |arg: &LinkArg| match arg {
            LinkArg::Const(c) => match &c.value {
                ConstValue::HostObject(h) => h.clone(),
                other => panic!("raise payload must be a HostObject Constant, got {other:?}"),
            },
            other => panic!("raise payload must be a Const link arg, got {other:?}"),
        };
        let w_type = host_of(&link.args[0]);
        assert!(w_type.is_class(), "w_type must be the exception class");
        assert!(w_type.qualname().contains("AssertionError"));
        let w_value = host_of(&link.args[1]);
        assert!(
            w_value.is_instance(),
            "w_value must be an exception instance, not the class"
        );
        assert_ne!(
            link.args[0], link.args[1],
            "exceptblock must not carry the same value in both slots"
        );
    }

    #[test]
    fn recloseblock_preserves_existing_exitswitch() {
        let mut graph = FunctionGraph::new("demo");
        let entry = graph.startblock;
        let cond_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "cond".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let target = graph.create_block();
        graph.block_mut(entry).exitswitch = Some(ExitSwitch::Value(cond_var.clone()));

        let link = Link::from_variables(&graph, vec![], target, None);
        graph.recloseblock(entry, vec![link]);

        assert_eq!(
            graph.block(entry).exitswitch,
            Some(ExitSwitch::Value(cond_var))
        );
        assert_eq!(graph.block(entry).exits[0].prevblock, Some(entry));
    }

    #[test]
    #[should_panic(expected = "already closed")]
    fn closeblock_panics_when_called_twice() {
        let mut graph = FunctionGraph::new("demo");
        let entry = graph.startblock;
        let first = graph.create_block();
        let second = graph.create_block();

        let first_link = Link::from_variables(&graph, vec![], first, None);
        let second_link = Link::from_variables(&graph, vec![], second, None);
        graph.closeblock(entry, vec![first_link]);
        graph.closeblock(entry, vec![second_link]);
    }

    #[test]
    fn framestate_union_kills_one_sided_local() {
        // RPython parity: `framestate.py:110-111` "if w1 is None or w2
        // is None: return None" — a slot present in one predecessor
        // but missing in the other is killed (dropped) from the merged
        // state.  Pyre realises this via dense positional entries:
        // position 0 (`x`) is bound on both sides but with distinct
        // Variable identities, position 1 (`self_only`) bound only on
        // self, position 2 (`other_only`) bound only on other, so both
        // one-sided slots collapse to None-kill at union.
        let mut graph = FunctionGraph::new("test");
        let x_self = graph.alloc_value_var();
        let self_only = graph.alloc_value_var();
        let x_other = graph.alloc_value_var();
        let other_only = graph.alloc_value_var();
        let self_state = FrameState {
            entries: vec![Some(x_self.clone()), Some(self_only), None],
            ..Default::default()
        };
        let other_state = FrameState {
            entries: vec![Some(x_other.clone()), None, Some(other_only)],
            ..Default::default()
        };
        let merged = self_state.union(&other_state, &mut graph).expect(
            "test invariant: AST frontend union is total — entries domain has no \
             UnionError, vestigial stack/exc/blocklist/next_offset (framestate.py:78)",
        );
        assert_eq!(
            merged.entries.len(),
            3,
            "merged state preserves positional length = max(len)"
        );
        let survivor = merged.entries[0]
            .as_ref()
            .expect("slot 0 (x) is bound on both sides and must survive");
        // Slot 0's predecessors disagreed on Variable identity, so
        // `union` minted a fresh phi Variable distinct from both.
        assert!(
            *survivor != x_self && *survivor != x_other,
            "disagreement must mint a fresh phi Variable distinct from both predecessors"
        );
        // Slots 1 (`self_only`) and 2 (`other_only`) are bound on
        // exactly one side each, so both collapse to None-kill —
        // upstream `framestate.py:110-111` "if w1 or w2 is None: …
        // return None" semantics.
        assert!(
            merged.entries[1].is_none() && merged.entries[2].is_none(),
            "one-sided slots must be None-killed positionally"
        );
    }

    /// Convenience helper: install an `OpKind::Input` op AND register its
    /// backing Variable as a phi inputarg on the same block.  Mirrors what
    /// the eager-phi installer at union sites does.
    fn install_phi(
        graph: &mut FunctionGraph,
        block: BlockId,
        name: &str,
    ) -> crate::flowspace::model::Variable {
        let phi_var = graph
            .push_op_var(
                block,
                OpKind::Input {
                    name: name.to_string(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(block, phi_var.clone());
        phi_var
    }

    #[test]
    fn prune_dead_phis_drops_orphan_phi_and_link_arg() {
        // entry → merge (phi 'x' never read) → returnblock(void)
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let const_v_var = graph
            .push_op_var(entry, OpKind::ConstInt(42), true)
            .unwrap();
        let merge = graph.create_block();
        install_phi(&mut graph, merge, "x");
        graph.set_goto(entry, merge, vec![const_v_var.clone()]);
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
    fn prune_dead_phis_retains_live_single_source_phi_pending_ssa_to_ssi() {
        // entry -> merge(phi 'x' read by a BinOp whose result is the
        // function return value) -> returnblock(reads return value).
        // RPython `remove_identical_vars_SSA` (simplify.py:561-563) would
        // collapse a phi whose incoming args are all the same value and rename
        // downstream readers to that representative, even a live reader.
        // `remove_duplicate_inputargs` omits that all-equal collapse
        // (PRE-EXISTING-ADAPTATION): on `crate::model` there is no `SSA_to_SSI`
        // repair pass, so the collapse would strand the reader on a value
        // defined in the predecessor and the flowspace adapter would reject it
        // as an undefined operand.  A *live* column is left threaded instead —
        // `SSA_to_SSI` would re-thread it regardless, so the collapse is a net
        // no-op.  This test pins the retained-phi shape; flip it back to the
        // collapse assertions once `SSA_to_SSI` is ported to `crate::model`.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let const_v_var = graph.push_op_var(entry, OpKind::ConstInt(7), true).unwrap();
        let merge = graph.create_block();
        let phi_x_var = install_phi(&mut graph, merge, "x");
        graph.set_goto(entry, merge, vec![const_v_var.clone()]);
        // BinOp whose result IS read (by `set_return`).  Backward
        // dataflow needs a live consumer of the BinOp result for the
        // pure-op-args→dependencies routing to keep phi_x alive.
        let doubled_var = graph
            .push_op_var(
                merge,
                OpKind::BinOp {
                    op: "int_add".into(),
                    lhs: phi_x_var.clone(),
                    rhs: phi_x_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(merge, Some(doubled_var.clone()));

        prune_dead_phis(&mut graph);

        assert_eq!(
            graph.block(merge).inputargs,
            vec![phi_x_var.clone()],
            "live single-source phi is retained (all-equal collapse deferred pending SSA_to_SSI)"
        );
        let entry_exit = &graph.block(entry).exits[0];
        assert_eq!(
            entry_exit.args.len(),
            1,
            "the predecessor link arg feeding the retained phi is kept"
        );
        let binop = graph
            .block(merge)
            .operations
            .iter()
            .find(|op| op.result.as_ref() == Some(&doubled_var))
            .expect("live BinOp remains");
        assert!(
            matches!(
                &binop.kind,
                OpKind::BinOp { lhs, rhs, .. } if lhs == &phi_x_var && rhs == &phi_x_var
            ),
            "live reader keeps reading the retained phi (not renamed)"
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
        let const_v_var = graph.push_op_var(entry, OpKind::ConstInt(7), true).unwrap();
        let merge = graph.create_block();
        let phi_x_var = install_phi(&mut graph, merge, "x");
        graph.set_goto(entry, merge, vec![const_v_var]);
        let doubled_var = graph
            .push_op_var(
                merge,
                OpKind::BinOp {
                    op: "int_add".into(),
                    lhs: phi_x_var.clone(),
                    rhs: phi_x_var,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        // Void return — `doubled_var` has no live consumer.
        graph.set_return(merge, None);

        prune_dead_phis(&mut graph);

        assert!(
            graph.block(merge).inputargs.is_empty(),
            "phi feeding only a dead pure op must collapse"
        );
        let has_binop = graph.block(merge).operations.iter().any(|op| {
            op.result.as_ref() == Some(&doubled_var) && matches!(op.kind, OpKind::BinOp { .. })
        });
        assert!(!has_binop, "dead pure op must be removed alongside the phi");
        let entry_exit = &graph.block(entry).exits[0];
        assert!(
            entry_exit.args.is_empty(),
            "predecessor link arg matching the pruned phi must be removed"
        );
    }

    #[test]
    fn remove_dead_aggregates_drops_discarded_ctor_and_its_field_stores() {
        // entry: tmp = SyntheticTransparentCtor("Tuple");
        //        setfield(tmp.__pos_0 = b0); setfield(tmp.__pos_1 = b1);
        //        return void.  `tmp` is never read, so the ctor and both
        // field stores are dead (`remove_simple_mallocs`).
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let b0 = graph
            .push_op_var(entry, OpKind::ConstBool(true), true)
            .unwrap();
        let b1 = graph
            .push_op_var(entry, OpKind::ConstBool(false), true)
            .unwrap();
        let tmp = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("Tuple"),
                    args: vec![],
                    result_ty: ValueType::Ref(Some("Tuple".into())),
                },
                true,
            )
            .unwrap();
        for (i, v) in [b0, b1].into_iter().enumerate() {
            graph.push_op_var(
                entry,
                OpKind::FieldWrite {
                    base: tmp.clone(),
                    field: FieldDescriptor {
                        name: format!("__pos_{i}"),
                        owner_root: Some("Tuple".into()),
                        owner_id: None,
                    },
                    value: LinkArg::Value(v),
                    ty: ValueType::Ref(None),
                },
                false,
            );
        }
        graph.set_return(entry, None);

        let removed = remove_dead_aggregates(&mut graph);

        assert_eq!(removed, 3, "ctor + two field stores must be removed");
        let has_ctor_or_store = graph.block(entry).operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { .. },
                    ..
                } | OpKind::FieldWrite { .. }
            )
        });
        assert!(
            !has_ctor_or_store,
            "dead aggregate ctor + stores must be gone"
        );
    }

    #[test]
    fn remove_dead_aggregates_keeps_returned_ctor() {
        // The same construction, but `tmp` is the return value — it is
        // read, so neither the ctor nor its stores may be removed.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let b0 = graph
            .push_op_var(entry, OpKind::ConstBool(true), true)
            .unwrap();
        let tmp = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("Tuple"),
                    args: vec![],
                    result_ty: ValueType::Ref(Some("Tuple".into())),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            entry,
            OpKind::FieldWrite {
                base: tmp.clone(),
                field: FieldDescriptor {
                    name: "__pos_0".into(),
                    owner_root: Some("Tuple".into()),
                    owner_id: None,
                },
                value: LinkArg::Value(b0),
                ty: ValueType::Ref(None),
            },
            false,
        );
        graph.set_return(entry, Some(tmp.clone()));

        let removed = remove_dead_aggregates(&mut graph);

        assert_eq!(removed, 0, "a returned aggregate must be kept");
        assert!(
            graph
                .block(entry)
                .operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&tmp)),
            "the live ctor op must survive"
        );
    }

    #[test]
    fn fuse_boxing_alloc_lowers_float_box_cluster() {
        // entry: %v       = const (the boxed payload);
        //        %header  = const (the ob_header value);
        //        %agg     = SyntheticTransparentCtor("W_FloatObject");
        //        FieldWrite(%agg.ob_header = %header);   // header store
        //        FieldWrite(%agg.floatval = %v);         // payload store
        //        %ret     = pyre_object.lltype.malloc_typed(%agg);
        //        return %ret.
        // fuse_boxing_alloc replaces the malloc with
        //        %ret     = NewWithVtable("W_FloatObject");
        //        FieldWrite(%ret.floatval = %v, ty=Float);
        // so the header rides the vtable descriptor (oracle: one NewWithVtable
        // + one floatval setfield, matching `codegen.rs trace_box_float`).
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let v = graph
            .push_op_var(entry, OpKind::ConstFloat(0.0f64.to_bits()), true)
            .unwrap();
        let header = graph.push_op_var(entry, OpKind::ConstInt(0), true).unwrap();
        let agg = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("W_FloatObject"),
                    args: vec![],
                    result_ty: ValueType::Ref(Some("W_FloatObject".into())),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            entry,
            OpKind::FieldWrite {
                base: agg.clone(),
                field: FieldDescriptor {
                    name: "ob_header".into(),
                    owner_root: Some("W_FloatObject".into()),
                    owner_id: None,
                },
                value: LinkArg::Value(header),
                ty: ValueType::Ref(None),
            },
            false,
        );
        graph.push_op_var(
            entry,
            OpKind::FieldWrite {
                base: agg.clone(),
                field: FieldDescriptor {
                    name: "floatval".into(),
                    owner_root: Some("W_FloatObject".into()),
                    owner_id: None,
                },
                value: LinkArg::Value(v.clone()),
                ty: ValueType::Ref(None),
            },
            false,
        );
        let ret = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "pyre_object".into(),
                            "lltype".into(),
                            "malloc_typed".into(),
                        ],
                    },
                    args: vec![agg.clone()],
                    result_ty: ValueType::Ref(Some("W_FloatObject".into())),
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(ret.clone()));

        let fused = fuse_boxing_alloc(&mut graph);
        assert_eq!(fused, 1, "exactly one boxing cluster must fuse");

        let ops = &graph.block(entry).operations;
        let nwv_pos = ops
            .iter()
            .position(|op| {
                matches!(&op.kind, OpKind::NewWithVtable { owner, .. } if owner == "W_FloatObject")
            })
            .expect("NewWithVtable must be emitted");
        assert_eq!(
            ops[nwv_pos].result.as_ref(),
            Some(&ret),
            "NewWithVtable must reuse the original malloc result register"
        );
        match &ops[nwv_pos + 1].kind {
            OpKind::FieldWrite {
                base, field, ty, ..
            } => {
                assert_eq!(
                    base, &ret,
                    "payload store must target the NewWithVtable result"
                );
                assert_eq!(field.name, "floatval", "payload field must be floatval");
                assert_eq!(
                    *ty,
                    ValueType::Float,
                    "payload store must be retyped to Float"
                );
            }
            other => panic!("expected payload FieldWrite after NewWithVtable, got {other:?}"),
        }
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments }, .. }
                    if segments.last().map(String::as_str) == Some("malloc_typed")
            )),
            "no malloc_typed call may survive the fusion"
        );
    }

    #[test]
    fn fuse_boxing_alloc_lowers_complex_box_two_payloads() {
        // A two-payload boxing struct (W_ComplexObject: real, imag) fuses to a
        // single NewWithVtable followed by two payload setfields in struct
        // order, both retyped to Float.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let re = graph
            .push_op_var(entry, OpKind::ConstFloat(0.0f64.to_bits()), true)
            .unwrap();
        let im = graph
            .push_op_var(entry, OpKind::ConstFloat(0.0f64.to_bits()), true)
            .unwrap();
        let agg = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("W_ComplexObject"),
                    args: vec![],
                    result_ty: ValueType::Ref(Some("W_ComplexObject".into())),
                },
                true,
            )
            .unwrap();
        for (name, v) in [("real", re.clone()), ("imag", im.clone())] {
            graph.push_op_var(
                entry,
                OpKind::FieldWrite {
                    base: agg.clone(),
                    field: FieldDescriptor {
                        name: name.into(),
                        owner_root: Some("W_ComplexObject".into()),
                        owner_id: None,
                    },
                    value: LinkArg::Value(v),
                    ty: ValueType::Ref(None),
                },
                false,
            );
        }
        let ret = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "pyre_object".into(),
                            "lltype".into(),
                            "malloc_typed".into(),
                        ],
                    },
                    args: vec![agg.clone()],
                    result_ty: ValueType::Ref(Some("W_ComplexObject".into())),
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(ret.clone()));

        let fused = fuse_boxing_alloc(&mut graph);
        assert_eq!(fused, 1, "the complex boxing cluster must fuse");

        let ops = &graph.block(entry).operations;
        let nwv_pos = ops
            .iter()
            .position(|op| {
                matches!(&op.kind, OpKind::NewWithVtable { owner, .. } if owner == "W_ComplexObject")
            })
            .expect("NewWithVtable must be emitted");
        // Two payload stores follow, real then imag, both targeting %ret/Float.
        let mut names = Vec::new();
        for off in 1..=2 {
            match &ops[nwv_pos + off].kind {
                OpKind::FieldWrite {
                    base, field, ty, ..
                } => {
                    assert_eq!(base, &ret, "payload store must target the alloc result");
                    assert_eq!(*ty, ValueType::Float, "complex payloads are Float");
                    names.push(field.name.clone());
                }
                other => panic!("expected payload FieldWrite, got {other:?}"),
            }
        }
        assert_eq!(
            names,
            vec!["real".to_string(), "imag".to_string()],
            "payload stores must follow struct order"
        );
    }

    #[test]
    fn fuse_boxing_alloc_ignores_unknown_owner() {
        // A malloc_typed of a non-boxing aggregate (no `payload_for` entry)
        // is left untouched — the fusion is opt-in per recognised struct.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let agg = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("SomeOtherStruct"),
                    args: vec![],
                    result_ty: ValueType::Ref(Some("SomeOtherStruct".into())),
                },
                true,
            )
            .unwrap();
        let ret = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "pyre_object".into(),
                            "lltype".into(),
                            "malloc_typed".into(),
                        ],
                    },
                    args: vec![agg.clone()],
                    result_ty: ValueType::Ref(Some("SomeOtherStruct".into())),
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(ret));

        let fused = fuse_boxing_alloc(&mut graph);
        assert_eq!(fused, 0, "unknown boxing owner must not fuse");
        assert!(
            !graph
                .block(entry)
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::NewWithVtable { .. })),
            "no NewWithVtable may be emitted for an unrecognised owner"
        );
    }

    #[test]
    fn fuse_boxing_alloc_sweeps_nested_header_chain() {
        // Faithful `w_float_new` shape: the boxing struct's header is a nested
        // `PyObject` ctor whose `ob_type` / `w_class` fields are fed by
        // `__pyre_cast_instance` pointer reinterprets of the `&FLOAT_TYPE`
        // constant.  After fusion drops the `ob_header` (it rides the
        // `NewWithVtable` descriptor), the entire header sub-tree — the inner
        // `PyObject` ctor, the outer `W_FloatObject` ctor, and the two
        // `__pyre_cast_instance` casts — is dead and must be swept, leaving
        // only the `NewWithVtable`, its `floatval` payload store, and the
        // return cast.  (The two `ConstRefAddr` constants legitimately survive
        // as dead constants; the dual-gate seeds them from the constant table,
        // so they do not diverge.)
        type Var = crate::flowspace::model::Variable;
        let cast_instance = |to: &str, arg: &Var| OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec!["__pyre_cast_instance".into(), to.into()],
            },
            args: vec![arg.clone()],
            result_ty: ValueType::Ref(Some(to.into())),
        };
        let field = |base: &Var, name: &str, owner: &str, value: &Var| OpKind::FieldWrite {
            base: base.clone(),
            field: FieldDescriptor {
                name: name.into(),
                owner_root: Some(owner.into()),
                owner_id: None,
            },
            value: LinkArg::Value(value.clone()),
            ty: ValueType::Ref(None),
        };
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let value = graph
            .push_op_var(entry, OpKind::ConstFloat(0.0f64.to_bits()), true)
            .unwrap();
        let ty_addr1 = graph
            .push_op_var(entry, OpKind::ConstRefAddr(4357049520), true)
            .unwrap();
        let ob_type = graph
            .push_op_var(entry, cast_instance("PyType", &ty_addr1), true)
            .unwrap();
        let ty_addr2 = graph
            .push_op_var(entry, OpKind::ConstRefAddr(4357049520), true)
            .unwrap();
        let w_class = graph
            .push_op_var(entry, cast_instance("PyType", &ty_addr2), true)
            .unwrap();
        // Inner `PyObject` header ctor + its field stores.
        let header = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("PyObject"),
                    args: vec![],
                    result_ty: ValueType::Ref(Some("PyObject".into())),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            entry,
            field(&header, "ob_type", "PyObject", &ob_type),
            false,
        );
        graph.push_op_var(
            entry,
            field(&header, "w_class", "PyObject", &w_class),
            false,
        );
        // Outer `W_FloatObject` ctor + its `ob_header` / `floatval` stores.
        let agg = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("W_FloatObject"),
                    args: vec![],
                    result_ty: ValueType::Ref(Some("W_FloatObject".into())),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            entry,
            field(&agg, "ob_header", "W_FloatObject", &header),
            false,
        );
        graph.push_op_var(
            entry,
            field(&agg, "floatval", "W_FloatObject", &value),
            false,
        );
        let raw = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "pyre_object".into(),
                            "lltype".into(),
                            "malloc_typed".into(),
                        ],
                    },
                    args: vec![agg.clone()],
                    result_ty: ValueType::Ref(Some("W_FloatObject".into())),
                },
                true,
            )
            .unwrap();
        let ret = graph
            .push_op_var(entry, cast_instance("PyObject", &raw), true)
            .unwrap();
        graph.set_return(entry, Some(ret.clone()));

        let fused = fuse_boxing_alloc(&mut graph);
        assert_eq!(fused, 1, "the nested-header boxing cluster must fuse");

        let ops = &graph.block(entry).operations;
        // The dead header sub-tree is gone: no SyntheticTransparentCtor and no
        // `__pyre_cast_instance["PyType"]` cast survives.
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { .. },
                    ..
                }
            )),
            "every dead aggregate ctor must be swept: {:#?}",
            ops.iter().map(|o| &o.kind).collect::<Vec<_>>()
        );
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments }, .. }
                    if segments.first().map(String::as_str) == Some("__pyre_cast_instance")
                        && segments.get(1).map(String::as_str) == Some("PyType")
            )),
            "the dead ob_type/w_class header casts must be swept"
        );
        // The live spine survives: NewWithVtable + floatval store + return cast.
        // The vtable (type pointer) is captured from the dropped
        // `ob_header.ob_type = cast(ConstRefAddr(4357049520))` store so the
        // runtime can stamp `ob_type` / `w_class`.
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::NewWithVtable { owner, vtable }
                    if owner == "W_FloatObject" && *vtable == 4357049520
            )),
            "NewWithVtable must survive carrying the captured type pointer"
        );
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments }, .. }
                    if segments.first().map(String::as_str) == Some("__pyre_cast_instance")
                        && segments.get(1).map(String::as_str) == Some("PyObject")
            )),
            "the live return cast must survive"
        );
        let _ = ret;
    }

    #[test]
    fn prune_dead_boxing_remnants_keeps_cast_used_as_live_store_base() {
        // The malloc-removal exemption (a store's base is not rooted) must be
        // scoped to fresh `SyntheticTransparentCtor` aggregates.  A store
        // *through* a `__pyre_cast_instance` alias is a real heap side effect:
        // even when the cast's result is read nowhere but the store base, the
        // cast and the store must survive.  The same graph carries a genuinely
        // dead `SyntheticTransparentCtor` whose store IS swept, so the pass is
        // proven to run rather than trivially early-returning.
        type Var = crate::flowspace::model::Variable;
        let field = |base: &Var, name: &str, value: &Var| OpKind::FieldWrite {
            base: base.clone(),
            field: FieldDescriptor {
                name: name.into(),
                owner_root: None,
                owner_id: None,
            },
            value: LinkArg::Value(value.clone()),
            ty: ValueType::Ref(None),
        };
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let obj = graph.alloc_value_var();
        graph.push_inputarg_var(entry, obj.clone());
        let payload = graph
            .push_op_var(entry, OpKind::ConstFloat(0.0f64.to_bits()), true)
            .unwrap();
        // A real mutation through a pointer reinterpret: `p` is read only as
        // the store base, yet the store must persist.
        let p = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["__pyre_cast_instance".into(), "W_FloatObject".into()],
                    },
                    args: vec![obj.clone()],
                    result_ty: ValueType::Ref(Some("W_FloatObject".into())),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(entry, field(&p, "floatval", &payload), false);
        // A genuinely dead fresh aggregate whose store is removable.
        let dead_agg = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("PyObject"),
                    args: vec![],
                    result_ty: ValueType::Ref(Some("PyObject".into())),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(entry, field(&dead_agg, "ob_type", &payload), false);
        graph.set_return(entry, Some(obj.clone()));

        prune_dead_boxing_remnants(&mut graph);

        let ops = &graph.block(entry).operations;
        assert!(
            ops.iter().any(|op| op.result.as_ref() == Some(&p)),
            "the cast aliasing a live store base must survive"
        );
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::FieldWrite { base, .. } if base == &p
            )),
            "the real store through the cast must survive"
        );
        assert!(
            !ops.iter().any(|op| op.result.as_ref() == Some(&dead_agg)),
            "the genuinely dead aggregate ctor must still be swept"
        );
    }

    #[test]
    fn prune_dead_boxing_remnants_sweeps_cross_block_threaded_cluster() {
        // The lowered `w_float_new` shape after fusion: the dead header cast
        // is produced in the entry block and threaded — by *reusing the same
        // `Variable`* as both the entry op result and the successor block's
        // inputarg (`set_goto_from_framestate` / `ensure_variable_at_block`) —
        // into the block where the dead `PyObject` header ctor reads it.  The
        // old `Link.arg`-counts-as-read sweep kept the whole cluster (the
        // threaded copy pinned the cast; the `FieldWrite` pinned the ctor);
        // the dependency-flow sweep reaches it because the threaded copy's
        // target inputarg is itself unread, then `prune_dead_phis` reclaims
        // the dangling inputarg / link arg / address constant.
        type Var = crate::flowspace::model::Variable;
        let cast_pytype = |arg: &Var| OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec!["__pyre_cast_instance".into(), "PyType".into()],
            },
            args: vec![arg.clone()],
            result_ty: ValueType::Ref(Some("PyType".into())),
        };
        let field = |base: &Var, name: &str, value: &Var| OpKind::FieldWrite {
            base: base.clone(),
            field: FieldDescriptor {
                name: name.into(),
                owner_root: Some("PyObject".into()),
                owner_id: None,
            },
            value: LinkArg::Value(value.clone()),
            ty: ValueType::Ref(None),
        };
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let value = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "value".into(),
                    ty: ValueType::Float,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(entry, value.clone());
        let get_instantiate = |arg: &Var| OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec![
                    "pyre_object".into(),
                    "pyobject".into(),
                    "get_instantiate".into(),
                ],
            },
            args: vec![arg.clone()],
            result_ty: ValueType::Ref(Some("object".into())),
        };
        let ty_addr1 = graph
            .push_op_var(entry, OpKind::ConstRefAddr(4357049520), true)
            .unwrap();
        // DEAD: the header `ob_type` cast, defined in the entry block.
        let cast = graph
            .push_op_var(entry, cast_pytype(&ty_addr1), true)
            .unwrap();
        let ty_addr2 = graph
            .push_op_var(entry, OpKind::ConstRefAddr(4357049520), true)
            .unwrap();
        // DEAD: a second cast feeding `get_instantiate`, whose result is the
        // header `w_class`.  `get_instantiate` is an observation-free
        // `instantiate`-slot load kept alive only by the threaded w_class
        // store, so it (and the cast feeding it) must be swept once the
        // header is dropped.
        let cast2 = graph
            .push_op_var(entry, cast_pytype(&ty_addr2), true)
            .unwrap();
        let inst = graph
            .push_op_var(entry, get_instantiate(&cast2), true)
            .unwrap();
        // LIVE: the fused boxing allocation + its payload store.
        let boxed = graph
            .push_op_var(
                entry,
                OpKind::NewWithVtable {
                    owner: "W_FloatObject".into(),
                    vtable: 0x5000_0000,
                },
                true,
            )
            .unwrap();
        graph.push_op_var(entry, field(&boxed, "floatval", &value), false);

        // Successor block reached after the boundary `get_instantiate` /
        // `malloc_typed` call would have opened; its inputargs reuse the SAME
        // `Variable`s the entry threads (id-reuse, not fresh phis).
        let blk = graph.create_block();
        graph.push_inputarg_var(blk, value.clone());
        graph.push_inputarg_var(blk, cast.clone());
        graph.push_inputarg_var(blk, inst.clone());
        graph.push_inputarg_var(blk, boxed.clone());
        graph.set_goto(
            entry,
            blk,
            vec![value.clone(), cast.clone(), inst.clone(), boxed.clone()],
        );

        // DEAD: the inner `PyObject` header ctor + its stores reading the cast
        // and the `get_instantiate` result.
        let header = graph
            .push_op_var(
                blk,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("PyObject"),
                    args: vec![],
                    result_ty: ValueType::Ref(Some("PyObject".into())),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(blk, field(&header, "ob_type", &cast), false);
        graph.push_op_var(blk, field(&header, "w_class", &inst), false);
        // LIVE: the return cast reads the boxing allocation.
        let ret = graph
            .push_op_var(
                blk,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["__pyre_cast_instance".into(), "PyObject".into()],
                    },
                    args: vec![boxed.clone()],
                    result_ty: ValueType::Ref(Some("PyObject".into())),
                },
                true,
            )
            .unwrap();
        graph.set_return(blk, Some(ret.clone()));

        prune_dead_boxing_remnants(&mut graph);
        prune_dead_phis(&mut graph);

        let kinds: Vec<&OpKind> = graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .map(|o| &o.kind)
            .collect();
        assert!(
            !kinds.iter().any(|k| matches!(
                k,
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { .. },
                    ..
                }
            )),
            "the cross-block dead header ctor must be swept: {kinds:#?}"
        );
        assert!(
            !kinds.iter().any(|k| matches!(
                k,
                OpKind::Call { target: CallTarget::FunctionPath { segments }, .. }
                    if segments.first().map(String::as_str) == Some("__pyre_cast_instance")
                        && segments.get(1).map(String::as_str) == Some("PyType")
            )),
            "the dead PyType cast threaded across the block boundary must be swept"
        );
        assert!(
            kinds.iter().any(
                |k| matches!(k, OpKind::NewWithVtable { owner, .. } if owner == "W_FloatObject")
            ),
            "the live NewWithVtable must survive"
        );
        assert!(
            kinds.iter().any(|k| matches!(
                k,
                OpKind::Call { target: CallTarget::FunctionPath { segments }, .. }
                    if segments.get(1).map(String::as_str) == Some("PyObject")
            )),
            "the live return cast must survive"
        );
        assert!(
            !kinds.iter().any(|k| matches!(
                k,
                OpKind::Call { target: CallTarget::FunctionPath { segments }, .. }
                    if segments.last().map(String::as_str) == Some("get_instantiate")
            )),
            "the dead get_instantiate (w_class feed) must be swept once the header is dropped"
        );
        assert!(
            !graph.block(blk).inputargs.contains(&cast)
                && !graph.block(blk).inputargs.contains(&inst),
            "the dangling threaded inputargs must be reclaimed by prune_dead_phis"
        );
        let _ = (ret, header, cast2, ty_addr1, ty_addr2);
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
        let const_v_var = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let merge1 = graph.create_block();
        let phi_x_var = install_phi(&mut graph, merge1, "x");
        graph.set_goto(entry, merge1, vec![const_v_var]);
        let merge2 = graph.create_block();
        install_phi(&mut graph, merge2, "y");
        graph.set_goto(merge1, merge2, vec![phi_x_var]);
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
        let unused_param_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "param".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(entry, unused_param_var.clone());
        graph.set_return(entry, None);

        let pre_returnblock_args = graph.block(graph.returnblock).inputargs.clone();
        let pre_exceptblock_args = graph.block(graph.exceptblock).inputargs.clone();

        prune_dead_phis(&mut graph);

        assert_eq!(
            graph.block(entry).inputargs,
            vec![unused_param_var.clone()],
            "startblock inputargs are calling-convention; never pruned"
        );
        assert_eq!(
            graph.block(graph.returnblock).inputargs.clone(),
            pre_returnblock_args,
            "returnblock inputargs are runtime contract; never pruned"
        );
        assert_eq!(
            graph.block(graph.exceptblock).inputargs.clone(),
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
        let lhs_var = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let rhs_var = graph.push_op_var(entry, OpKind::ConstInt(2), true).unwrap();
        let raising_var = graph
            .push_op_var(
                entry,
                OpKind::BinOp {
                    op: "int_add".into(),
                    lhs: lhs_var,
                    rhs: rhs_var,
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
        let etype_var = graph.push_op_var(entry, OpKind::ConstInt(0), true).unwrap();
        let evalue_var = graph.push_op_var(entry, OpKind::ConstInt(0), true).unwrap();
        let normal_arg = LinkArg::Value(raising_var.clone());
        let exc_etype_arg = LinkArg::Value(etype_var);
        let exc_evalue_arg = LinkArg::Value(evalue_var);
        {
            let block = graph.block_mut(entry);
            block.exitswitch = Some(ExitSwitch::LastException);
            block.exits = vec![
                Link::new_mixed(vec![normal_arg], returnblock, None),
                Link::new_mixed(vec![exc_etype_arg, exc_evalue_arg], exceptblock, None),
            ];
        }

        prune_dead_phis(&mut graph);

        let still_present = graph
            .block(entry)
            .operations
            .iter()
            .any(|op| op.result.as_ref() == Some(&raising_var));
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
        let unused_param_var = graph
            .push_op_var(
                orphan_entry,
                OpKind::Input {
                    name: "captured".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(orphan_entry, unused_param_var.clone());

        prune_dead_phis(&mut graph);

        assert_eq!(
            graph.block(orphan_entry).inputargs,
            vec![unused_param_var.clone()],
            "non-canonical entry block (no incoming link) must keep its inputargs"
        );
    }

    /// `FrameState::union` walks the merged stack.  The disagreement at
    /// stack slot 0 (Variable vs Variable with distinct identities)
    /// yields a fresh phi Variable per `framestate.py:113-114 return
    /// Variable()`; that fresh identity is distinct from both
    /// predecessors.
    #[test]
    fn union_registers_fresh_stack_phi_variable_with_valueid() {
        use crate::flowspace::framestate::StackElem;
        use crate::flowspace::model::{Hlvalue, Variable};

        let mut graph = FunctionGraph::new("union_stack_phi");
        let v_a = Variable::new();
        let v_b = Variable::new();

        let pred_a = FrameState {
            entries: Vec::new(),
            stack: vec![StackElem::Value(Hlvalue::Variable(v_a.clone()))],
            ..Default::default()
        };
        let pred_b = FrameState {
            entries: Vec::new(),
            stack: vec![StackElem::Value(Hlvalue::Variable(v_b.clone()))],
            ..Default::default()
        };

        let merged = pred_a
            .union(&pred_b, &mut graph)
            .expect("disagreeing-Variable union must succeed");
        let StackElem::Value(Hlvalue::Variable(phi)) = &merged.stack[0] else {
            panic!("disagreement must mint a fresh phi Variable");
        };
        assert!(
            *phi != v_a && *phi != v_b,
            "fresh phi Variable must be distinct from both predecessors after union"
        );
    }

    /// Carry-through stack Variable (identical Hlvalue identity on both
    /// predecessors) carries through to the same Variable handle without
    /// minting a fresh phi — `framestate.py:108 if w1 == w2: return w1`
    /// parity.
    #[test]
    fn union_stack_carry_through_variable_keeps_existing_valueid() {
        use crate::flowspace::framestate::StackElem;
        use crate::flowspace::model::{Hlvalue, Variable};

        let mut graph = FunctionGraph::new("union_stack_carry");
        let shared = Variable::new();

        let pred_a = FrameState {
            entries: Vec::new(),
            stack: vec![StackElem::Value(Hlvalue::Variable(shared.clone()))],
            ..Default::default()
        };
        let pred_b = FrameState {
            entries: Vec::new(),
            stack: vec![StackElem::Value(Hlvalue::Variable(shared.clone()))],
            ..Default::default()
        };

        let merged = pred_a
            .union(&pred_b, &mut graph)
            .expect("carry-through union must succeed");
        let StackElem::Value(Hlvalue::Variable(carried)) = &merged.stack[0] else {
            panic!("Variable carry-through expected");
        };
        assert_eq!(
            carried, &shared,
            "carry-through must reuse the shared Variable identity"
        );
    }

    /// `FSException` projections route through the same registration
    /// walk: a fresh phi Variable minted by the per-cell exception
    /// `union` lands in `merged.last_exception` with a paired slot.
    #[test]
    fn union_registers_fresh_exception_phi_variable_with_valueid() {
        use crate::flowspace::model::{FSException, Hlvalue, Variable};

        let mut graph = FunctionGraph::new("union_exc_phi");
        let v_t1 = Variable::new();
        let v_t2 = Variable::new();
        let v_v1 = Variable::new();
        let v_v2 = Variable::new();
        let pred_a = FrameState {
            entries: Vec::new(),
            stack: Vec::new(),
            last_exception: Some(FSException::new(
                Hlvalue::Variable(v_t1.clone()),
                Hlvalue::Variable(v_v1.clone()),
            )),
            ..Default::default()
        };
        let pred_b = FrameState {
            entries: Vec::new(),
            stack: Vec::new(),
            last_exception: Some(FSException::new(
                Hlvalue::Variable(v_t2.clone()),
                Hlvalue::Variable(v_v2.clone()),
            )),
            ..Default::default()
        };

        let merged = pred_a
            .union(&pred_b, &mut graph)
            .expect("disagreeing-exception union must succeed");
        let exc = merged
            .last_exception
            .as_ref()
            .expect("merged must carry FSException");
        let Hlvalue::Variable(v_t_phi) = &exc.w_type else {
            panic!("w_type phi expected");
        };
        let Hlvalue::Variable(v_v_phi) = &exc.w_value else {
            panic!("w_value phi expected");
        };
        assert!(
            *v_t_phi != v_t1 && *v_t_phi != v_t2,
            "w_type disagreement must mint a fresh phi Variable"
        );
        assert!(
            *v_v_phi != v_v1 && *v_v_phi != v_v2,
            "w_value disagreement must mint a fresh phi Variable"
        );
    }

    /// `FrameState::union` derives the parallel
    /// `locals_w` (`Hlvalue` carrier matching `framestate.py:19
    /// self.locals_w`) from the unioned `entries` (Variable carrier).
    /// Each defined slot's Variable carries through directly via
    /// clone; None-killed slots stay None.
    #[test]
    fn union_derives_locals_w_hlvalue_carrier_from_entries() {
        use crate::flowspace::model::{Hlvalue, Variable};
        let mut graph = FunctionGraph::new("locals_w_derive");
        let v_shared = Variable::new();
        let v_a_only = Variable::new();
        let v_b_only = Variable::new();

        let pred_a = FrameState {
            entries: vec![Some(v_shared.clone()), Some(v_a_only.clone()), None],
            ..Default::default()
        };
        let pred_b = FrameState {
            entries: vec![Some(v_shared.clone()), None, Some(v_b_only.clone())],
            ..Default::default()
        };
        let merged = pred_a
            .union(&pred_b, &mut graph)
            .expect("union must succeed");
        assert_eq!(
            merged.entries.len(),
            3,
            "entries carrier preserves max-extension length",
        );
        assert_eq!(
            merged.locals_w.len(),
            merged.entries.len(),
            "locals_w must be in lockstep with entries",
        );
        // Slot 0: agreement at shared_vid → carry-through, locals_w[0]
        // is the same Variable identity.
        assert!(
            matches!(&merged.locals_w[0], Some(Hlvalue::Variable(v)) if v.id() == v_shared.id()),
            "carry-through slot 0 must mirror shared Variable identity",
        );
        // Slots 1, 2: None-killed at union, locals_w mirrors None.
        assert!(
            merged.locals_w[1].is_none() && merged.locals_w[2].is_none(),
            "None-killed slots stay None in locals_w",
        );
    }

    /// `set_goto_from_framestate` threads `getoutputargs`'s
    /// mixed-shape result through `Link::new_mixed` per
    /// `flowcontext.py:438 block.recloseblock(Link(outputargs,
    /// newblock))`.  A predecessor with one locals slot + one stack
    /// Constant + one stack Variable yields a Link whose `args`
    /// carry both `LinkArg::Value` (locals + stack Variable) and
    /// `LinkArg::Const` (stack Constant) side by side.
    #[test]
    fn set_goto_from_framestate_emits_mixed_link_args_through_new_mixed() {
        use crate::flowspace::framestate::StackElem;
        use crate::flowspace::model::{Constant, Hlvalue, Variable};

        let mut graph = FunctionGraph::new("set_goto_mixed");
        let pred = graph.create_block();

        let v_local = Variable::new();
        let v_stack = Variable::new();

        let pred_state = FrameState {
            entries: vec![Some(v_local.clone())],
            stack: vec![
                StackElem::Value(Hlvalue::Constant(Constant::new(ConstValue::Int(42)))),
                StackElem::Value(Hlvalue::Variable(v_stack)),
            ],
            ..Default::default()
        };
        // Construct a target FrameState whose mergeable projection
        // demands all three cells via `Variable` placeholders (locals
        // slot is Some, stack cells are Hlvalue::Variable).
        let v_target_stack_0 = Variable::new();
        let v_target_stack_1 = Variable::new();
        let target_state = FrameState {
            entries: vec![Some(v_local.clone())],
            stack: vec![
                StackElem::Value(Hlvalue::Variable(v_target_stack_0)),
                StackElem::Value(Hlvalue::Variable(v_target_stack_1)),
            ],
            ..Default::default()
        };
        // SpamBlock(framestate) parity — inputargs derive from
        // target_state.getvariables() so the `simplify.py:513` invariant
        // `len(link.args) == len(target.inputargs)` is satisfied by
        // construction.
        let merge = graph.create_block_from_framestate(&target_state);

        graph.set_goto_from_framestate(pred, merge, &pred_state, &target_state);
        let pred_meta = graph.block(pred);
        assert_eq!(pred_meta.exits.len(), 1, "single-exit Link expected");
        let link = &pred_meta.exits[0];
        assert_eq!(link.target, merge);
        assert_eq!(link.args.len(), 3, "locals + stack const + stack variable");
        assert!(
            matches!(&link.args[0], LinkArg::Value(v) if v.id() == v_local.id()),
            "locals projection must carry the local Variable identity",
        );
        assert!(
            matches!(&link.args[1], LinkArg::Const(c) if matches!(c.value, ConstValue::Int(42))),
            "stack Constant must route to LinkArg::Const directly",
        );
        assert!(
            matches!(&link.args[2], LinkArg::Value(_)),
            "stack Variable must route to LinkArg::Value",
        );
    }

    /// `flowspace/model.py:114 Link.__init__` asserts
    /// `len(args) == len(target.inputargs)` at link construction.  The
    /// pyre port enforces the same predicate inside
    /// `set_goto_from_framestate` so a framestate-driven recloseblock
    /// catches a mergeable-shape mismatch before the link is wired.
    #[test]
    #[should_panic(expected = "outputargs.len() (1) != target.inputargs.len() (0)")]
    fn set_goto_from_framestate_panics_on_inputarg_length_mismatch() {
        use crate::flowspace::model::Variable;
        let mut graph = FunctionGraph::new("inputarg_len_assert");
        let pred = graph.create_block();
        // Target block created without inputargs — empty inputargs vec.
        let target = graph.create_block();
        let v_local = Variable::new();
        let pred_state = FrameState {
            entries: vec![Some(v_local.clone())],
            ..Default::default()
        };
        // pred_state.mergeable Variable cells: 1 locals.
        // target_state.mergeable Variable cells: 1 locals.
        // getoutputargs returns 1 LinkArg, but target.inputargs is empty.
        let target_state = FrameState {
            entries: vec![Some(v_local.clone())],
            ..Default::default()
        };
        graph.set_goto_from_framestate(pred, target, &pred_state, &target_state);
    }

    /// `ensure_variable_at_block` short-circuits when the Variable is
    /// already in `block.inputargs` — no graph mutation, returns true.
    #[test]
    fn ensure_variable_at_block_idempotent_on_existing_inputarg() {
        use crate::flowspace::model::Variable;
        let mut graph = FunctionGraph::new("ensure_var_idempotent_input");
        let block = graph.create_block();
        let v = Variable::new();
        graph.block_mut(block).inputargs.push(v.clone());

        let inputargs_before = graph.block(block).inputargs.len();
        let ok = graph.ensure_variable_at_block(block, &v);
        assert!(ok, "Variable already in inputargs reports success");
        assert_eq!(
            graph.block(block).inputargs.len(),
            inputargs_before,
            "no duplicate inputarg insertion",
        );
    }

    /// `ensure_variable_at_block` short-circuits when the Variable is
    /// already the result of an op in `block` — definition-by-op-result.
    #[test]
    fn ensure_variable_at_block_idempotent_on_existing_op_result() {
        let mut graph = FunctionGraph::new("ensure_var_idempotent_op");
        let block = graph.create_block();
        let _v = graph.push_op_var(
            block,
            OpKind::Input {
                name: "x".to_string(),
                ty: ValueType::Unknown,
                class_root: None,
            },
            true,
        );
        let var = graph.block(block).operations[0]
            .result
            .as_ref()
            .unwrap()
            .clone();

        let inputargs_before = graph.block(block).inputargs.len();
        let ok = graph.ensure_variable_at_block(block, &var);
        assert!(ok, "op-result definition reports success");
        assert_eq!(
            graph.block(block).inputargs.len(),
            inputargs_before,
            "no inputarg insertion when op result already defines var",
        );
    }

    /// `ensure_variable_at_block` returns false when `block` has no
    /// predecessors AND `var` is not defined locally — the malformed
    /// graph case callers should treat as a bug.
    #[test]
    fn ensure_variable_at_block_returns_false_on_orphan_block_without_definition() {
        use crate::flowspace::model::Variable;
        let mut graph = FunctionGraph::new("ensure_var_orphan");
        let orphan = graph.create_block();
        let v = Variable::new();
        let ok = graph.ensure_variable_at_block(orphan, &v);
        assert!(!ok, "no predecessors + not defined => fails (no threading)");
        assert!(
            graph.block(orphan).inputargs.is_empty(),
            "failure path performs no mutation",
        );
    }

    /// `ensure_variable_at_block` threads a Variable defined in a
    /// predecessor through one intermediate block: appends to the
    /// predecessor's exit args + adds to the block's inputargs.
    /// Mirrors the carry-through case in `flowspace/flowcontext.py:407
    /// setstate(block.framestate)` at block entry.
    #[test]
    fn ensure_variable_at_block_threads_one_level_through_predecessor() {
        use crate::flowspace::model::Variable;
        let mut graph = FunctionGraph::new("ensure_var_one_level");
        let pred = graph.create_block();
        let target = graph.create_block();
        let v = Variable::new();
        // Define v in pred via an Input op result.
        let _ = graph.push_op_var(
            pred,
            OpKind::Input {
                name: "p".to_string(),
                ty: ValueType::Unknown,
                class_root: None,
            },
            true,
        );
        // Override the op result Variable to `v` so the definition
        // predicate matches.
        graph.block_mut(pred).operations[0].result = Some(v.clone());
        graph.set_goto(pred, target, vec![]);

        let ok = graph.ensure_variable_at_block(target, &v);
        assert!(ok, "predecessor defines v => threading succeeds");
        assert_eq!(
            graph.block(target).inputargs.len(),
            1,
            "target gains v as inputarg",
        );
        assert!(graph.block(target).inputargs.contains(&v));
        assert_eq!(
            graph.block(pred).exits[0].args.len(),
            1,
            "pred exit args grow by one entry",
        );
        assert!(
            matches!(&graph.block(pred).exits[0].args[0], LinkArg::Value(var) if var.id() == v.id()),
            "pred exit args carry the same Variable identity",
        );
    }

    /// `ensure_variable_at_block` recurses through two levels: the
    /// intermediate block also gains the inputarg and the deepest
    /// predecessor's exit args grow.
    #[test]
    fn ensure_variable_at_block_threads_two_levels_through_predecessor_chain() {
        use crate::flowspace::model::Variable;
        let mut graph = FunctionGraph::new("ensure_var_two_level");
        let root = graph.create_block();
        let mid = graph.create_block();
        let leaf = graph.create_block();
        let v = Variable::new();
        let _ = graph.push_op_var(
            root,
            OpKind::Input {
                name: "p".to_string(),
                ty: ValueType::Unknown,
                class_root: None,
            },
            true,
        );
        graph.block_mut(root).operations[0].result = Some(v.clone());
        graph.set_goto(root, mid, vec![]);
        graph.set_goto(mid, leaf, vec![]);

        let ok = graph.ensure_variable_at_block(leaf, &v);
        assert!(ok, "two-level chain threading succeeds");
        assert!(
            graph.block(leaf).inputargs.contains(&v),
            "leaf gains v inputarg",
        );
        assert!(
            graph.block(mid).inputargs.contains(&v),
            "mid gains v inputarg",
        );
        assert_eq!(
            graph.block(root).exits[0].args.len(),
            1,
            "root exit args grow",
        );
        assert_eq!(
            graph.block(mid).exits[0].args.len(),
            1,
            "mid exit args grow",
        );
    }

    /// Cycle-safe: a back-edge predecessor finds the in-progress
    /// inputarg slot via `variable_defined_in_block` and short-circuits.
    /// Loop shape: `header → body_tail → header`.
    #[test]
    fn ensure_variable_at_block_handles_back_edge_without_infinite_recursion() {
        use crate::flowspace::model::Variable;
        let mut graph = FunctionGraph::new("ensure_var_back_edge");
        let entry = graph.create_block();
        let header = graph.create_block();
        let body_tail = graph.create_block();
        let v = Variable::new();
        let _ = graph.push_op_var(
            entry,
            OpKind::Input {
                name: "p".to_string(),
                ty: ValueType::Unknown,
                class_root: None,
            },
            true,
        );
        graph.block_mut(entry).operations[0].result = Some(v.clone());
        graph.set_goto(entry, header, vec![]);
        graph.set_goto(header, body_tail, vec![]);
        graph.set_goto(body_tail, header, vec![]);

        let ok = graph.ensure_variable_at_block(header, &v);
        assert!(ok, "back-edge cycle is handled");
        assert!(
            graph.block(header).inputargs.contains(&v),
            "header inputarg seeded",
        );
        assert!(
            graph.block(body_tail).inputargs.contains(&v),
            "body_tail inputarg also seeded — back-edge predecessor demanded v",
        );
        assert_eq!(
            graph.block(entry).exits[0].args.len(),
            1,
            "entry → header link args extended",
        );
        assert_eq!(
            graph.block(body_tail).exits[0].args.len(),
            1,
            "body_tail → header link args extended",
        );
        assert_eq!(
            graph.block(header).exits[0].args.len(),
            1,
            "header → body_tail link args extended (body_tail demands v)",
        );
    }

    /// `framestate.py:50-51 getvariables` walks `locals + flatten(stack) +
    /// [exc_type, exc_value]` in order and filters Variable cells.  The
    /// Pyre port emits Variables from each projection in the same
    /// positional order so `block.inputargs` line up with
    /// `getoutputargs` cells.
    #[test]
    fn getvariables_walks_mergeable_in_locals_stack_exc_order() {
        use crate::flowspace::framestate::StackElem;
        use crate::flowspace::model::{Constant, FSException, Hlvalue, Variable};

        let graph = FunctionGraph::new("getvariables_shape");
        let v_local = Variable::new();
        let v_stack = Variable::new();
        let v_exc_type = Variable::new();
        let v_exc_value = Variable::new();

        let fs = FrameState {
            entries: vec![Some(v_local.clone()), None],
            stack: vec![
                StackElem::Value(Hlvalue::Constant(Constant::new(ConstValue::Int(7)))),
                StackElem::Value(Hlvalue::Variable(v_stack.clone())),
            ],
            last_exception: Some(FSException::new(
                Hlvalue::Variable(v_exc_type.clone()),
                Hlvalue::Variable(v_exc_value.clone()),
            )),
            ..Default::default()
        };
        let vars = fs.getvariables(&graph);
        assert_eq!(
            vars.len(),
            4,
            "1 locals + 1 stack Variable + 2 exc Variables"
        );
        assert_eq!(vars[0].id(), v_local.id(), "locals slot first");
        assert_eq!(
            vars[1].id(),
            v_stack.id(),
            "stack Variable second (Constant filtered)"
        );
        assert_eq!(vars[2].id(), v_exc_type.id(), "exc_type third");
        assert_eq!(vars[3].id(), v_exc_value.id(), "exc_value fourth");
    }

    /// `create_block_from_framestate` materialises the merge block
    /// shape that `flowcontext.py:38 SpamBlock(framestate)` builds —
    /// `block.inputargs = framestate.getvariables()`, attached
    /// `block.framestate = framestate`.  Inputarg Variables carry the
    /// framestate's Variable identities directly.
    #[test]
    fn create_block_from_framestate_sets_inputargs_and_attaches_state() {
        use crate::flowspace::framestate::StackElem;
        use crate::flowspace::model::{Hlvalue, Variable};

        let mut graph = FunctionGraph::new("spamblock_shape");
        let v_local = Variable::new();
        let v_stack = Variable::new();

        let fs = FrameState {
            entries: vec![Some(v_local.clone())],
            stack: vec![StackElem::Value(Hlvalue::Variable(v_stack.clone()))],
            ..Default::default()
        };
        let merge = graph.create_block_from_framestate(&fs);
        let block = graph.block(merge);
        assert_eq!(
            block.inputargs.len(),
            2,
            "1 locals + 1 stack Variable (no exception)"
        );
        assert_eq!(block.inputargs[0].id(), v_local.id());
        assert_eq!(block.inputargs[1].id(), v_stack.id());
        assert!(
            block.framestate.is_some(),
            "block.framestate stamped (SpamBlock parity)"
        );
        assert_eq!(&block.inputargs[0], &v_local);
        assert_eq!(&block.inputargs[1], &v_stack);
    }

    /// The critical round-trip invariant: predecessor's
    /// `getoutputargs(target=merged_state)` length matches
    /// `create_block_from_framestate(merged_state).inputargs.len()` —
    /// the simplify.py:513 invariant `len(link.args) ==
    /// len(target.inputargs)`.  Demonstrated against a non-trivial
    /// mergeable shape (locals + stack Variable + stack Constant +
    /// exception cells) so the locals/stack/exc walk all participate.
    #[test]
    fn outputargs_and_inputargs_round_trip_for_full_mergeable_shape() {
        use crate::flowspace::framestate::StackElem;
        use crate::flowspace::model::{Constant, FSException, Hlvalue, Variable};

        let mut graph = FunctionGraph::new("mergeable_round_trip");
        let v_a = Variable::new();
        let v_b = Variable::new();

        // Two predecessors disagreeing on stack slot 0 (Variable phi)
        // and on exception (Variable phi); locals agree on v_a.
        let v_t1 = Variable::new();
        let v_t2 = Variable::new();
        let v_v1 = Variable::new();
        let v_v2 = Variable::new();
        let pred_a = FrameState {
            entries: vec![Some(v_a.clone())],
            stack: vec![
                StackElem::Value(Hlvalue::Variable(v_b.clone())),
                StackElem::Value(Hlvalue::Constant(Constant::new(ConstValue::Int(1)))),
            ],
            last_exception: Some(FSException::new(
                Hlvalue::Variable(v_t1),
                Hlvalue::Variable(v_v1),
            )),
            ..Default::default()
        };
        let pred_b = FrameState {
            entries: vec![Some(v_a.clone())],
            stack: vec![
                StackElem::Value(Hlvalue::Variable(v_b.clone())),
                StackElem::Value(Hlvalue::Constant(Constant::new(ConstValue::Int(1)))),
            ],
            last_exception: Some(FSException::new(
                Hlvalue::Variable(v_t2),
                Hlvalue::Variable(v_v2),
            )),
            ..Default::default()
        };
        let merged = pred_a
            .union(&pred_b, &mut graph)
            .expect("union must succeed");
        // Construct the merge block from the merged framestate; then
        // ask each predecessor for its outputargs against the same
        // target.  The shapes must line up positionally — the
        // simplify.py:513 invariant.
        let merge_block_id = graph.create_block_from_framestate(&merged);
        let merge_block = graph.block(merge_block_id);
        let outputargs_a = pred_a.getoutputargs(&merged, &graph);
        let outputargs_b = pred_b.getoutputargs(&merged, &graph);
        assert_eq!(
            outputargs_a.len(),
            merge_block.inputargs.len(),
            "pred_a outputargs length matches merge block inputargs",
        );
        assert_eq!(
            outputargs_b.len(),
            merge_block.inputargs.len(),
            "pred_b outputargs length matches merge block inputargs",
        );
    }

    /// `try_getoutputargs` returns `None` (rather than panicking like
    /// `getoutputargs`) when the target binds a `Variable` at a locals
    /// slot the predecessor leaves `None` — the phantom-slot mismatch the
    /// cyclic framestate path's loop-header pre-seed can produce when a
    /// live-in phi is scrubbed undefined on one edge.  The all-bound case
    /// still returns `Some`.
    #[test]
    fn try_getoutputargs_declines_phantom_slot_returns_some_when_bound() {
        use crate::flowspace::model::Variable;

        let graph = FunctionGraph::new("try_getoutputargs_phantom");
        // Target binds a Variable phi at slot 0 (a loop-header live-in).
        let target = FrameState {
            entries: vec![Some(Variable::new())],
            ..Default::default()
        };
        // Predecessor whose slot 0 was scrubbed to `None` (undefined on
        // this edge): the target Variable slot has no self cell to thread.
        let pred_undefined = FrameState {
            entries: vec![None],
            ..Default::default()
        };
        assert!(
            pred_undefined.try_getoutputargs(&target, &graph).is_none(),
            "phantom slot 0 undefined in predecessor declines to None",
        );
        // Predecessor that does bind slot 0 threads it through.
        let v_self = Variable::new();
        let pred_bound = FrameState {
            entries: vec![Some(v_self.clone())],
            ..Default::default()
        };
        let out = pred_bound
            .try_getoutputargs(&target, &graph)
            .expect("bound slot threads through");
        assert_eq!(out, vec![LinkArg::Value(v_self)]);
    }

    // ── annotator-monomorphization Slice C2 — CallTarget::Method ──
    // transient ClassDefKey transport invariants.

    #[test]
    fn call_target_method_default_resolved_path_is_none() {
        let t = CallTarget::method("foo", Some("Bar".to_string()));
        assert_eq!(t.resolved_path(), None);
    }

    #[test]
    fn call_target_method_resolved_path_is_skipped_by_serde() {
        use crate::parse::CallPath;
        let path = CallPath::for_impl_method("PyFrame", "foo");
        let t = CallTarget::Method {
            name: "foo".into(),
            receiver_root: Some("Bar".into()),
            resolved_path: Some(path),
        };
        let json = serde_json::to_string(&t).expect("encode");
        assert!(
            !json.contains("resolved_path"),
            "serialized form leaks resolved_path: {json}"
        );
        let round_trip: CallTarget = serde_json::from_str(&json).expect("decode");
        assert_eq!(round_trip.resolved_path(), None);
    }
}
