//! Charon ULLBC schema (basic-block CFG form).
//!
//! Mirrors the subset of Charon 0.1.196's ULLBC the lowering driver
//! consumes. The layout is reverse-engineered from
//! `majit/charon-corpus/corpus.ullbc`; see
//! `majit/charon-corpus/README.md` for the schema findings.
//!
//! ## Schema-drift policy
//!
//! - Fields we **read** are typed (`Ty`, `StmtKind`, `TermKind`, …).
//! - Fields we do not yet read stay as [`serde_json::Value`]. This is
//!   forward-compatible: a new Charon release that adds fields here
//!   loads without code changes.
//! - Enum variants we know about are listed by name. Unknown variants
//!   are surfaced via `#[serde(other)]` arms named `Unknown` so the
//!   reader fails-loud at the lowering site instead of silently
//!   discarding work.

use serde::Deserialize;
use serde_json::Value;
use serde_json::value::RawValue;

// ---------------------------------------------------------------------------
// FunDecl + meta
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct FunDecl {
    pub def_id: u64,
    pub item_meta: ItemMeta,
    pub signature: Signature,
    /// Generic-parameter context of the declaration: `types` lists the
    /// type params (`{"index": N, "name": "H"|"Self"}`) and
    /// `trait_clauses` their bounds
    /// (`trait_.skip_binder.{id, generics.types[0]}` = bound trait id +
    /// subject TypeVar).  Kept as raw `Value`; only
    /// `front::mir::tyref_generic_trait_bound_root` projects it, to map
    /// a `&T`-where-`T: Trait` parameter to its bound trait's name leaf.
    #[serde(default)]
    pub generics: Option<Value>,
    /// Charon stamps this with the `GlobalDecl` id when the function
    /// is a compiler-synthesised static / const initialiser body
    /// (e.g. the body that constructs `static NONE_SINGLETON`'s
    /// value).  Production lowering treats these as values rather
    /// than call targets — they have no call sites in user code, and
    /// their unwind paths use orphan exception slots that the
    /// flowspace adapter cannot lift.  `None` for ordinary function
    /// bodies.
    #[serde(default)]
    pub is_global_initializer: Option<u64>,
    /// `body` is `null` for opaque references and one of
    /// `{"Unstructured": {...}}`, `{"Structured": {...}}`, or
    /// `{"Error": {...}}` otherwise. Kept as the raw JSON text (not an
    /// exploded `Value` tree) so the whole-corpus retained footprint
    /// stays near the on-disk size; project to `Unstructured` on demand
    /// via [`FunDecl::unstructured`]. A schema change in the unused
    /// variants still does not break load.
    pub body: Option<Box<RawValue>>,
}

impl FunDecl {
    /// Return the `Unstructured` (basic-block CFG) body if present.
    pub fn unstructured(&self) -> Option<Unstructured> {
        #[derive(Deserialize)]
        struct Proj {
            #[serde(rename = "Unstructured")]
            unstructured: Unstructured,
        }
        let body = self.body.as_ref()?;
        serde_json::from_str::<Proj>(body.get())
            .ok()
            .map(|p| p.unstructured)
    }

    /// Source name of the first argument local (local index 1; index 0 is the
    /// return slot).  `None` when the body is absent, the function has no
    /// arguments, or Charon dropped the name.
    ///
    /// Projects only the `locals` table — the basic-block CFG is skipped — so a
    /// caller can distinguish a `self` receiver from an associated
    /// constructor's `Self`-typed parameter without paying for a full body
    /// parse.  Rust forbids naming a parameter `self`, so a genuine receiver's
    /// local is the `self` keyword (named `"self"`, or unnamed for a
    /// monomorphised trait method), whereas `W_Range::allocate(payload: Self)`
    /// keeps its source name (`"payload"`).
    pub fn first_arg_local_name(&self) -> Option<String> {
        #[derive(Deserialize)]
        struct Proj {
            #[serde(rename = "Unstructured")]
            u: BodyProj,
        }
        #[derive(Deserialize)]
        struct BodyProj {
            locals: LocalsProj,
        }
        #[derive(Deserialize)]
        struct LocalsProj {
            arg_count: u64,
            locals: Vec<LocalNameProj>,
        }
        #[derive(Deserialize)]
        struct LocalNameProj {
            #[serde(default)]
            name: Option<String>,
        }
        let body = self.body.as_ref()?;
        let proj = serde_json::from_str::<Proj>(body.get()).ok()?;
        if proj.u.locals.arg_count < 1 {
            return None;
        }
        proj.u.locals.locals.get(1).and_then(|l| l.name.clone())
    }

    /// Returns `Some(msg)` if Charon recorded a translation error
    /// (e.g. `"charon does not support thread local references"`).
    pub fn error_message(&self) -> Option<String> {
        #[derive(Deserialize)]
        struct Proj {
            #[serde(rename = "Error")]
            error: ErrBody,
        }
        #[derive(Deserialize)]
        struct ErrBody {
            msg: String,
        }
        let body = self.body.as_ref()?;
        serde_json::from_str::<Proj>(body.get())
            .ok()
            .map(|p| p.error.msg)
    }
}

/// Static or const item referenced via [`PlaceKind::Global`].
///
/// Body / type / initialiser remain opaque (`Value`); the only field
/// the lowering driver consumes is [`ItemMeta::name_path`] for
/// constructing a stable `CallTarget::FunctionPath`-style identifier.
#[derive(Debug, Deserialize)]
pub struct GlobalDecl {
    pub def_id: u64,
    pub item_meta: ItemMeta,
    #[serde(flatten)]
    pub rest: std::collections::BTreeMap<String, Value>,
}

/// User-defined type (`struct` / `enum` / `type` alias / opaque
/// forward-decl) the program references. The `kind` field is consumed
/// to populate `SemanticProgram.{known_struct_names,
/// struct_fields, known_trait_names}` from the LLBC alone.
#[derive(Debug, Deserialize)]
pub struct TypeDecl {
    pub def_id: u64,
    pub item_meta: ItemMeta,
    pub kind: TypeDeclKind,
    /// Per-target memory layout Charon resolved for the concrete type:
    /// a list of `{key: <target-triple>, value: TypeLayout}` entries
    /// (one per extraction target — a single-target extraction carries
    /// one). Field byte offsets live in
    /// `value.variant_layouts[variant_idx].field_offsets` (a struct is a
    /// single variant); the discriminant tag position in
    /// `value.discriminator`. Kept as raw JSON text and projected on
    /// demand via [`TypeDecl::layout_for_target`] so an unmodelled or
    /// exotic layout shape can never break the whole-LLBC load (the
    /// schema-drift policy above). `None` when Charon emitted no layout.
    #[serde(default)]
    pub layout: Option<Box<RawValue>>,
}

/// One `{key, value}` entry of [`TypeDecl::layout`].
#[derive(Debug, Deserialize)]
pub struct TargetLayout {
    pub key: String,
    pub value: TypeLayout,
}

/// Resolved memory layout of a single concrete type for one target.
#[derive(Debug, Deserialize)]
pub struct TypeLayout {
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default)]
    pub align: Option<u64>,
    /// One entry per variant (a struct has a single entry). Carries the
    /// per-variant field byte offsets.
    #[serde(default)]
    pub variant_layouts: Vec<VariantLayout>,
    /// Tag/niche decoder (`{"Branch": {"offset": ..., ...}}` for a
    /// niche/tag enum). Kept raw; read the tag byte position via
    /// [`TypeLayout::discriminant_offset`].
    #[serde(default)]
    pub discriminator: Option<Value>,
}

/// Per-variant field offsets within [`TypeLayout`].
#[derive(Debug, Deserialize)]
pub struct VariantLayout {
    /// Byte offset of each field, indexed by the variant's field
    /// declaration order (matches [`VariantDecl::fields`] / a struct's
    /// [`FieldDecl`] order).
    #[serde(default)]
    pub field_offsets: Vec<u64>,
}

impl TypeDecl {
    /// Project the per-target layout, selecting the entry whose target
    /// triple equals `target`, or the sole entry when exactly one is
    /// present (single-target extraction). Returns `None` when layout is
    /// absent, unparseable, or no entry matches — callers fall back to
    /// the heuristic layout provider.
    pub fn layout_for_target(&self, target: &str) -> Option<TypeLayout> {
        let raw = self.layout.as_ref()?;
        let entries: Vec<TargetLayout> = serde_json::from_str(raw.get()).ok()?;
        select_target_layout(entries, target)
    }
}

/// Pick the [`TargetLayout`] matching `target`, or the sole entry when
/// exactly one is present. Returns `None` for an empty list or a
/// multi-target list with no match.
fn select_target_layout(mut entries: Vec<TargetLayout>, target: &str) -> Option<TypeLayout> {
    if let Some(pos) = entries.iter().position(|e| e.key == target) {
        return Some(entries.swap_remove(pos).value);
    }
    if entries.len() == 1 {
        return Some(entries.swap_remove(0).value);
    }
    None
}

impl TypeLayout {
    /// Byte offset of field `field_idx` within variant `variant_idx`.
    pub fn field_offset(&self, variant_idx: usize, field_idx: usize) -> Option<u64> {
        self.variant_layouts
            .get(variant_idx)?
            .field_offsets
            .get(field_idx)
            .copied()
    }

    /// Byte offset of field `field_idx` of a struct (the single variant).
    pub fn struct_field_offset(&self, field_idx: usize) -> Option<u64> {
        self.field_offset(0, field_idx)
    }

    /// Byte position of the discriminant tag (`discriminator.Branch.offset`).
    /// `None` for a single-variant type or a non-`Branch` discriminator.
    pub fn discriminant_offset(&self) -> Option<u64> {
        self.discriminator
            .as_ref()?
            .get("Branch")?
            .get("offset")?
            .as_u64()
    }
}

#[derive(Debug, Deserialize)]
pub enum TypeDeclKind {
    /// Struct body — vector of field declarations.
    Struct(Vec<FieldDecl>),
    /// Enum body — vector of variant declarations. Each variant
    /// carries its own field list (zero-arg for unit variants, named
    /// for `Foo { a: ... }`, positional for `Bar(T)`).
    Enum(Vec<VariantDecl>),
    /// Union body — a field list shaped like `Struct`'s, but every field
    /// starts at offset 0 and only one is live at a time. Foreign C unions
    /// reach the table through the dependency closure (`windows-sys`'
    /// `IN_ADDR_0`), so the variant must parse even though no field-offset
    /// row can be projected from it.
    Union(Vec<FieldDecl>),
    /// Type alias (`type T = ...`). The aliased type lives in
    /// `rest["aliased_ty"]`; not currently consumed.
    Alias(Value),
    /// Forward declaration / opaque type (Charon couldn't see body).
    Opaque,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
pub struct FieldDecl {
    pub name: Option<String>,
    pub ty: TyRef,
    #[serde(default)]
    pub attr_info: Option<AttrInfo>,
}

#[derive(Debug, Deserialize)]
pub struct VariantDecl {
    pub name: String,
    #[serde(default)]
    pub fields: Vec<FieldDecl>,
    /// Charon-assigned discriminant, kept raw because its scalar width
    /// varies by enum (`{"Scalar":{"Unsigned":["U8","128"]}}` for
    /// `Instruction`, `{"Scalar":{"Signed":["Isize","0"]}}` for others).
    /// Read via [`VariantDecl::discriminant_i64`]; staying [`Value`]
    /// keeps deserialization total under the schema-drift policy.
    #[serde(default)]
    pub discriminant: Option<Value>,
}

impl VariantDecl {
    /// Parse the discriminant to `i64` from the Charon
    /// `{"Scalar":{"Signed"|"Unsigned":[width, decimal_string]}}` shape.
    /// Returns `None` for an absent, non-scalar, or unparseable
    /// discriminant rather than failing — callers that need the value
    /// for an enum known to carry integer discriminants assert presence
    /// at the use site.
    pub fn discriminant_i64(&self) -> Option<i64> {
        let scalar = self.discriminant.as_ref()?.get("Scalar")?;
        let pair = scalar.get("Unsigned").or_else(|| scalar.get("Signed"))?;
        pair.get(1)?.as_str()?.parse::<i64>().ok()
    }
}

/// Trait declaration — referenced when populating
/// `SemanticProgram.known_trait_names`. Body intentionally minimal:
/// only `item_meta.name_path()` is consumed.
#[derive(Debug, Deserialize)]
pub struct TraitDecl {
    pub def_id: u64,
    pub item_meta: ItemMeta,
}

#[derive(Debug, Deserialize)]
pub struct ItemMeta {
    pub name: Vec<NameSeg>,
    pub span: Span,
    pub source_text: Option<String>,
    pub attr_info: AttrInfo,
    #[serde(default)]
    pub is_local: bool,
}

impl ItemMeta {
    /// `"crate::module::item"`-style flattened name. Trait-impl
    /// segments and other non-ident segments are rendered as
    /// `"<Variant>"`.
    pub fn name_path(&self) -> String {
        let mut out = String::new();
        for (i, seg) in self.name.iter().enumerate() {
            if i > 0 {
                out.push_str("::");
            }
            match seg {
                NameSeg::Ident {
                    ident: (s, disambiguator),
                } => {
                    out.push_str(s);
                    // Multiple anonymous closures in one function all
                    // carry the bare segment `closure`; only the
                    // disambiguator index distinguishes them.  Keep it
                    // (as `closure#N`) so co-located closure envs mint
                    // distinct ClassDefs instead of collapsing their
                    // captured `__pos_N` fields onto one shared row.
                    if s == "closure" && *disambiguator > 0 {
                        out.push('#');
                        out.push_str(&disambiguator.to_string());
                    }
                }
                NameSeg::Other(v) => {
                    let label = v
                        .as_object()
                        .and_then(|m| m.keys().next().cloned())
                        .unwrap_or_else(|| "?".into());
                    out.push('<');
                    out.push_str(&label);
                    out.push('>');
                }
            }
        }
        out
    }
}

/// Whether a struct-root leaf names a closure env — the bare `closure`
/// or a disambiguated `closure#N` (see [`ItemMeta::name_path`]).  Leaf
/// checks that special-case closures must accept both spellings.
pub fn is_closure_leaf(leaf: &str) -> bool {
    leaf == "closure" || leaf.starts_with("closure#")
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum NameSeg {
    Ident {
        #[serde(rename = "Ident")]
        ident: (String, u64),
    },
    Other(Value),
}

#[derive(Debug, Deserialize)]
pub struct AttrInfo {
    pub attributes: Vec<Value>,
    /// `"Hint"` / `"Always"` / `"Never"` for explicit `#[inline*]` ;
    /// `null` for functions without any inline attribute.
    pub inline: Option<String>,
    pub rename: Option<String>,
    pub public: bool,
}

#[derive(Debug, Deserialize)]
pub struct Span {
    pub data: SpanData,
}

#[derive(Debug, Deserialize)]
pub struct SpanData {
    pub file_id: u64,
    pub beg: Loc,
    pub end: Loc,
}

#[derive(Debug, Deserialize)]
pub struct Loc {
    pub line: u64,
    pub col: u64,
}

#[derive(Debug, Deserialize)]
pub struct Signature {
    pub is_unsafe: bool,
    pub inputs: Vec<TyRef>,
    pub output: TyRef,
}

// ---------------------------------------------------------------------------
// Types — kept thin. The lowering driver only needs to *label* types
// for diff output, not deeply reason about them, so the type table is
// not walked here.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum TyRef {
    /// Pointer into the global type-dedup table.
    Dedup {
        #[serde(rename = "Deduplicated")]
        id: u64,
    },
    /// Inline value with hash-cons id.
    Inline {
        #[serde(rename = "HashConsedValue")]
        value: (u64, Value),
    },
    /// Anything else (e.g. literal-int short forms).
    Other(Value),
}

impl TyRef {
    /// Stable single-line label, e.g. `"ty#170"` or `"ty<Adt>"`.
    pub fn label(&self) -> String {
        match self {
            TyRef::Dedup { id } => format!("ty#{id}"),
            TyRef::Inline { value: (id, _) } => format!("ty#{id}*"),
            TyRef::Other(v) => v
                .as_object()
                .and_then(|m| m.keys().next().cloned())
                .map(|k| format!("ty<{k}>"))
                .unwrap_or_else(|| "ty<?>".into()),
        }
    }
}

// ---------------------------------------------------------------------------
// Bodies
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct Unstructured {
    pub locals: Locals,
    pub body: Vec<BasicBlock>,
    pub span: Span,
}

#[derive(Debug, Deserialize)]
pub struct Locals {
    pub arg_count: u64,
    pub locals: Vec<Local>,
}

#[derive(Debug, Deserialize)]
pub struct Local {
    pub index: u64,
    pub name: Option<String>,
    pub span: Span,
    pub ty: TyRef,
}

#[derive(Debug, Deserialize)]
pub struct BasicBlock {
    pub statements: Vec<Statement>,
    /// Raw terminator. Project to [`TermKind`] via [`BasicBlock::term`]
    /// so a parse error on a single terminator does not poison the
    /// whole function.
    pub terminator: Value,
}

impl BasicBlock {
    /// Project the terminator into the typed [`TermKind`] enum.
    /// Returns the raw JSON in the error if a variant is unknown so
    /// callers can decide whether to fail-loud or fall back.
    pub fn term(&self) -> Result<TermKind, String> {
        let kind = self
            .terminator
            .as_object()
            .and_then(|m| m.get("kind"))
            .ok_or_else(|| "terminator has no 'kind'".to_string())?;
        serde_json::from_value(kind.clone()).map_err(|e| format!("{e}; raw kind: {kind}"))
    }
}

#[derive(Debug, Deserialize)]
pub struct Statement {
    /// Raw statement-kind JSON.
    pub kind: Value,
    pub span: Span,
}

impl Statement {
    /// Project to the typed [`StmtKind`] enum.
    pub fn stmt_kind(&self) -> Result<StmtKind, String> {
        serde_json::from_value::<StmtKind>(self.kind.clone())
            .map_err(|e| format!("{e}; raw kind: {}", self.kind))
    }
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub enum StmtKind {
    /// Local enters scope.
    StorageLive(u64),
    /// Local leaves scope.
    StorageDead(u64),
    /// `place := rvalue`
    Assign(Place, Rvalue),
    /// `Assert { cond, expected, check_kind }` — inline assertion;
    /// failure terminator is the *terminator-level* `Assert` instead.
    Assert(AssertStmt),
    /// `let _ = place` style references (MIR `PlaceMention`).
    PlaceMention(Place),
    /// Anything else (e.g. `Deinit`, `SetDiscriminant`, …).
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
pub struct AssertStmt {
    pub cond: Operand,
    pub expected: bool,
    pub check_kind: Value,
}

// ---------------------------------------------------------------------------
// Places, operands, rvalues
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct Place {
    pub kind: PlaceKind,
    pub ty: TyRef,
}

#[derive(Debug, Deserialize)]
pub enum PlaceKind {
    Local(u64),
    Projection(Box<Place>, ProjectionElem),
    /// Reference to a static / const global item.
    /// `Global { generics, id }` — `id` indexes `global_decls`.
    Global {
        generics: Value,
        id: u64,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ProjectionElem {
    /// `"Deref"` and similar atom variants.
    Atom(String),
    Tagged(Value),
}

impl ProjectionElem {
    pub fn label(&self) -> String {
        match self {
            ProjectionElem::Atom(s) => s.clone(),
            ProjectionElem::Tagged(v) => {
                if let Some(obj) = v.as_object()
                    && let Some(k) = obj.keys().next()
                {
                    return k.clone();
                }
                "?".into()
            }
        }
    }
}

#[derive(Debug, Deserialize)]
pub enum Rvalue {
    Use(Operand),
    /// `BinaryOp(op, lhs, rhs)`. `op` is a tagged variant — primitive
    /// ops are atom strings (`"Add"`, `"Eq"`, …), wrap/overflow forms
    /// are objects (`{"Shr": "Wrap"}`, `{"Add": "Wrap"}`).
    BinaryOp(Value, Operand, Operand),
    UnaryOp(Value, Operand),
    /// `Ref { place, kind, ptr_metadata }` — borrow / raw-ptr creation.
    Ref {
        place: Place,
        /// `"Shared" | "Mut" | "TwoPhaseMut" | …`
        kind: Value,
        ptr_metadata: Value,
    },
    /// `Aggregate(kind, operands)` — tuple / struct / enum-variant /
    /// array construction.
    Aggregate(Value, Vec<Operand>),
    Discriminant(Place),
    /// `Cast(kind, operand, target_ty)`.
    Cast(Value, Operand, TyRef),
    /// `Len(place)` for slice / array length.
    Len(Place),
    /// `Repeat(operand, elem_ty, count)` for `[v; N]` literals.
    Repeat(Operand, TyRef, Value),
    /// `ShallowInitBox(operand, target_ty)` — emitted by `Box::new_in`
    /// and friends to allocate the box and initialise its contents.
    ShallowInitBox(Operand, TyRef),
    /// `RawPtr { place, kind }` — raw-pointer construction (sibling of `Ref`).
    RawPtr {
        place: Place,
        kind: Value,
        ptr_metadata: Value,
    },
    /// `NullaryOp(op, type)` — `SizeOf(T)`, `AlignOf(T)`, etc.
    NullaryOp(Value, TyRef),
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
pub enum Operand {
    Copy(Place),
    Move(Place),
    Const(Value),
}

// ---------------------------------------------------------------------------
// Terminators
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub enum TermKind {
    Return,
    UnwindResume,
    Abort(Value),
    Goto {
        target: u64,
    },
    Switch {
        discr: Operand,
        targets: SwitchTargets,
    },
    Call {
        call: CallPayload,
        target: u64,
        on_unwind: u64,
    },
    Assert {
        assert: AssertStmt,
        target: u64,
        on_unwind: u64,
    },
    Drop {
        target: u64,
        on_unwind: u64,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
pub enum SwitchTargets {
    /// Boolean switch: `[then_bb, else_bb]`.
    If(u64, u64),
    /// `SwitchInt(int_ty, [(scalar, bb)], default_bb)`.
    SwitchInt(Value, Vec<(Value, u64)>, u64),
}

#[derive(Debug, Deserialize)]
pub struct CallPayload {
    pub func: CallFunc,
    pub args: Vec<Operand>,
    pub dest: Place,
}

/// Charon's `Call.func` is one of two top-level variants:
///   - `Regular { kind, generics }`   — statically resolved
///   - `Dynamic <operand>`            — `dyn Trait` virtual call
///
/// The inner `kind` of `Regular` further distinguishes `Fun(Regular n)`
/// (monomorphized direct call), `Fun(Trait …)` (trait-bound generic
/// resolved at extraction time), or `Ptr` (function-pointer call).
#[derive(Debug, Deserialize)]
pub enum CallFunc {
    Regular(RegularCall),
    /// `dyn Trait` virtual call. The operand carries the fat pointer
    /// the dispatch reads from.
    Dynamic(Operand),
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
pub struct RegularCall {
    pub kind: CallKind,
    pub generics: Value,
}

#[derive(Debug, Deserialize)]
pub enum CallKind {
    /// Statically resolved function call: `Fun { Regular(fn_id) }` or
    /// `Fun { Trait(...) }`.
    Fun(FunId),
    /// Static trait method call (post-resolution).
    Trait(Value),
    /// `Ptr` (function-pointer call).
    Ptr(Value),
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum FunId {
    Regular {
        #[serde(rename = "Regular")]
        id: u64,
    },
    Other(Value),
}

impl CallFunc {
    /// Bucket the call into a `CallClass` for the lowering driver to
    /// dispatch on.
    pub fn classify(&self) -> CallClass {
        match self {
            CallFunc::Regular(r) => match &r.kind {
                CallKind::Fun(FunId::Regular { .. }) => CallClass::Direct,
                CallKind::Fun(FunId::Other(_)) | CallKind::Trait(_) => CallClass::Trait,
                CallKind::Ptr(_) => CallClass::Ptr,
                CallKind::Unknown => CallClass::Unknown,
            },
            CallFunc::Dynamic(_) => CallClass::Dynamic,
            CallFunc::Unknown => CallClass::Unknown,
        }
    }
}

/// Bucket the lowering driver dispatches on for call terminators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallClass {
    /// Monomorphized direct call — emit as `direct_call`.
    Direct,
    /// Trait-bound generic — also direct (Charon already resolved).
    Trait,
    /// `dyn Trait` virtual call — emit as indirect call through fat
    /// pointer; lowering may devirtualize on type-flow.
    Dynamic,
    /// Function-pointer call.
    Ptr,
    /// Unrecognised — fail-loud at lowering site.
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A struct's field offsets are the single `variant_layouts[0]` entry.
    #[test]
    fn struct_layout_field_offsets() {
        let json = r#"{
            "size": 80, "align": 8,
            "discriminator": null, "uninhabited": false,
            "variant_layouts": [
                {"field_offsets": [0, 24, 48, 72, 73], "uninhabited": false, "tagger": []}
            ],
            "repr": {"repr_algo": "Rust"}
        }"#;
        let layout: TypeLayout = serde_json::from_str(json).unwrap();
        assert_eq!(layout.size, Some(80));
        assert_eq!(layout.align, Some(8));
        assert_eq!(layout.struct_field_offset(0), Some(0));
        assert_eq!(layout.struct_field_offset(3), Some(72));
        assert_eq!(layout.struct_field_offset(4), Some(73));
        assert_eq!(layout.struct_field_offset(5), None);
        // single-variant type has no decodable discriminant tag
        assert_eq!(layout.discriminant_offset(), None);
    }

    /// An enum's per-variant field offsets, plus the niche tag position.
    #[test]
    fn enum_layout_per_variant_offsets_and_tag() {
        let json = r#"{
            "size": 104, "align": 8,
            "discriminator": {"Branch": {"offset": 0, "int_ty": {"Unsigned": "U8"}}},
            "uninhabited": false,
            "variant_layouts": [
                {"field_offsets": [0], "uninhabited": false, "tagger": []},
                {"field_offsets": [8], "uninhabited": false,
                 "tagger": [[0, {"Unsigned": ["U8", "6"]}]]}
            ],
            "repr": {"repr_algo": "Rust"}
        }"#;
        let layout: TypeLayout = serde_json::from_str(json).unwrap();
        assert_eq!(layout.field_offset(0, 0), Some(0));
        assert_eq!(layout.field_offset(1, 0), Some(8));
        assert_eq!(layout.field_offset(2, 0), None);
        assert_eq!(layout.discriminant_offset(), Some(0));
    }

    /// Target selection: exact-match wins; a sole entry is chosen even
    /// when the requested target differs; an empty/no-match list is `None`.
    #[test]
    fn target_layout_selection() {
        let two = r#"[
            {"key": "x86_64-unknown-linux-gnu",
             "value": {"variant_layouts": [{"field_offsets": [0]}]}},
            {"key": "aarch64-apple-darwin",
             "value": {"size": 16, "variant_layouts": [{"field_offsets": [8]}]}}
        ]"#;
        let entries: Vec<TargetLayout> = serde_json::from_str(two).unwrap();
        // exact match selects the aarch64 entry (offset 8), not the first
        assert_eq!(
            select_target_layout(entries, "aarch64-apple-darwin")
                .and_then(|l| l.struct_field_offset(0)),
            Some(8)
        );

        let sole = r#"[{"key": "x86_64-unknown-linux-gnu",
            "value": {"variant_layouts": [{"field_offsets": [0]}]}}]"#;
        let entries: Vec<TargetLayout> = serde_json::from_str(sole).unwrap();
        // sole entry chosen despite a non-matching requested target
        assert_eq!(
            select_target_layout(entries, "aarch64-apple-darwin")
                .and_then(|l| l.struct_field_offset(0)),
            Some(0)
        );

        // empty list → None
        assert!(select_target_layout(Vec::new(), "aarch64-apple-darwin").is_none());
    }

    /// A union body keeps its field list. `#[serde(other)]` turns any tag this
    /// enum does not name into `Unknown`, so a missing `Union` arm would lose
    /// the fields silently rather than fail to parse — the shape `IN_ADDR_0`
    /// arrives in through the `windows-sys` dependency closure.
    #[test]
    fn union_decl_kind_keeps_its_fields() {
        let json = r#"{"Union": [
            {"name": "S_un_b", "ty": {"Deduplicated": 170}, "attr_info": null},
            {"name": "S_addr", "ty": {"Deduplicated": 42}, "attr_info": null}
        ]}"#;
        let kind: TypeDeclKind = serde_json::from_str(json).unwrap();
        let TypeDeclKind::Union(fields) = kind else {
            panic!("union body parsed as {kind:?}");
        };
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name.as_deref(), Some("S_un_b"));
        assert_eq!(fields[1].ty.label(), "ty#42");
    }
}
