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
use std::sync::OnceLock;

// FunDecl + meta

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
    /// Where the function comes from (`ItemSource`): `"Normal"`,
    /// `{"TraitImpl": ..}`, `{"GlobalInitializer": {"id": <GlobalDecl id>,
    /// ..}}`, …  Kept as raw `Value`; only
    /// [`FunDecl::is_global_initializer`] projects it.
    #[serde(default)]
    pub src: Option<Value>,
    /// `body` is `null` for opaque references and one of
    /// `{"Unstructured": {...}}`, `{"Structured": {...}}`, or
    /// `{"Error": {...}}` otherwise. Kept as the raw JSON text (not an
    /// exploded `Value` tree) so the whole-corpus retained footprint
    /// stays near the on-disk size; project to `Unstructured` on demand
    /// via [`FunDecl::unstructured`]. A schema change in the unused
    /// variants still does not break load.
    pub body: Option<Box<RawValue>>,
    /// [`FunDecl::has_unstructured_body`], decided on first query.
    #[serde(skip)]
    has_unstructured_body: OnceLock<bool>,
    /// [`FunDecl::first_arg_local_name`], decided on first query.
    #[serde(skip)]
    first_arg_local_name: OnceLock<Option<String>>,
}

impl FunDecl {
    /// The `GlobalDecl` id when the function is a compiler-synthesised
    /// static / const initialiser body (e.g. the body that constructs
    /// `static NONE_SINGLETON`'s value), read from
    /// `src: {"GlobalInitializer": {"id": ..}}`.  Production lowering
    /// treats these as values rather than call targets — they have no
    /// call sites in user code, and their unwind paths use orphan
    /// exception slots that the flowspace adapter cannot lift.  `None`
    /// for ordinary function bodies.
    pub fn is_global_initializer(&self) -> Option<u64> {
        self.src
            .as_ref()?
            .get("GlobalInitializer")?
            .get("id")?
            .as_u64()
    }

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

    /// Whether the body is the `Unstructured` variant, the one
    /// [`Self::unstructured`] projects.
    ///
    /// Call lowering asks this of a callee at every call site; the answer is
    /// a property of the declaration, so it is decided once, from the
    /// variant tag alone: the basic blocks are skipped, not built.
    pub fn has_unstructured_body(&self) -> bool {
        *self.has_unstructured_body.get_or_init(|| {
            #[derive(Deserialize)]
            struct Proj {
                #[serde(rename = "Unstructured")]
                _unstructured: serde::de::IgnoredAny,
            }
            self.body
                .as_ref()
                .is_some_and(|body| serde_json::from_str::<Proj>(body.get()).is_ok())
        })
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
        self.first_arg_local_name
            .get_or_init(|| self.project_first_arg_local_name())
            .clone()
    }

    fn project_first_arg_local_name(&self) -> Option<String> {
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
    /// Origin Charon recorded for this declaration. A compiler-generated
    /// closure environment is the object `{"Closure": …}`; an ordinary ADT
    /// is the string `"Normal"`. Kept raw so an
    /// unmodelled origin cannot fail the load.
    #[serde(default)]
    pub src: Option<Value>,
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
    /// Byte size. Charon spells this as a bare integer or as
    /// `{"chosen": <u64 | const expr>}`; a `Deduplicated` chosen value is
    /// filled in by [`TypeDecl::layout_for_target`].
    #[serde(default, deserialize_with = "de_layout_opt_u64")]
    pub size: Option<u64>,
    #[serde(default, deserialize_with = "de_layout_opt_u64")]
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
    #[serde(default)]
    pub repr: Option<TypeRepr>,
}

/// Representation properties that affect the low-level value shape.
#[derive(Debug, Deserialize)]
pub struct TypeRepr {
    #[serde(default)]
    pub transparent: bool,
}

/// Per-variant field offsets within [`TypeLayout`].
#[derive(Debug, Deserialize)]
pub struct VariantLayout {
    /// Byte offset of each field, indexed by the variant's field
    /// declaration order (matches [`VariantDecl::fields`] / a struct's
    /// [`FieldDecl`] order). Each entry is a bare integer or
    /// `{"chosen": <u64>}`.
    #[serde(default, deserialize_with = "de_layout_offsets")]
    pub field_offsets: Vec<u64>,
    /// The writes that store this variant's tag, as
    /// `[[offset, {"Unsigned": ["U8", "<value>"]}], ...]`. Kept raw; read
    /// via [`VariantLayout::tag_i64`].
    #[serde(default)]
    pub tagger: Option<Value>,
}

impl VariantLayout {
    /// The tag value this variant stores, when the tagger is a single
    /// scalar write.
    pub fn tag_i64(&self) -> Option<i64> {
        let [write] = self.tagger.as_ref()?.as_array()?.as_slice() else {
            return None;
        };
        let scalar = write.as_array()?.get(1)?;
        let pair = scalar.get("Unsigned").or_else(|| scalar.get("Signed"))?;
        pair.get(1)?.as_str()?.parse::<i64>().ok()
    }
}

impl TypeDecl {
    /// Project the per-target layout, selecting the entry whose target
    /// triple equals `target`, or the sole entry when exactly one is
    /// present (single-target extraction). Returns `None` when layout is
    /// absent, unparseable, or no entry matches — callers fall back to
    /// the heuristic layout provider.
    ///
    /// `size` / `align` whose `chosen` value is a `Deduplicated` constant
    /// are resolved through `llbc`'s const table.
    pub fn layout_for_target(&self, llbc: &crate::Llbc, target: &str) -> Option<TypeLayout> {
        let raw = self.layout.as_ref()?;
        let entries: Vec<TargetLayout> = serde_json::from_str(raw.get()).ok()?;
        let mut layout = select_target_layout(entries, target)?;
        if layout.size.is_none() {
            layout.size = layout_measure(llbc, raw.get(), target, "size");
        }
        if layout.align.is_none() {
            layout.align = layout_measure(llbc, raw.get(), target, "align");
        }
        Some(layout)
    }

    /// Whether every emitted target layout records `#[repr(transparent)]`.
    pub fn is_repr_transparent(&self) -> bool {
        let Some(raw) = self.layout.as_ref() else {
            return false;
        };
        let Ok(entries) = serde_json::from_str::<Vec<TargetLayout>>(raw.get()) else {
            return false;
        };
        !entries.is_empty()
            && entries.iter().all(|entry| {
                entry
                    .value
                    .repr
                    .as_ref()
                    .is_some_and(|repr| repr.transparent)
            })
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

/// A layout integer written inline: a bare `u64`, `{"chosen": <u64>}`, or a
/// constant expression whose literal is `{"Integer": {"Unsigned"|"Signed":
/// [width, decimal]}}`. A `Deduplicated` chosen value stays unresolved so
/// [`layout_measure`] can read it from the const table.
fn layout_u64_literal(v: &Value) -> Option<u64> {
    if let Some(n) = v.as_u64() {
        return Some(n);
    }
    if let Some(chosen) = v.get("chosen") {
        if chosen.get("Deduplicated").is_some() {
            return None;
        }
        return layout_u64_literal(chosen);
    }
    if let Some(body) = v
        .get("Value")
        .and_then(Value::as_array)
        .and_then(|a| a.get(1))
    {
        return layout_u64_literal(body);
    }
    if let Some(constant) = v.get("Constant") {
        return layout_u64_literal(constant);
    }
    if let Some(int) = v.get("Integer") {
        let pair = int.get("Unsigned").or_else(|| int.get("Signed"))?;
        return pair.get(1)?.as_str()?.parse().ok();
    }
    v.as_array()
        .and_then(|a| a.first())
        .and_then(layout_u64_literal)
}

fn de_layout_opt_u64<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<u64>, D::Error> {
    let value = Option::<Value>::deserialize(d)?;
    Ok(value.as_ref().and_then(layout_u64_literal))
}

fn de_layout_offsets<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<u64>, D::Error> {
    let values = Vec::<Value>::deserialize(d)?;
    values
        .iter()
        .map(|v| {
            layout_u64_literal(v).ok_or_else(|| {
                serde::de::Error::custom(format!("field offset is not a literal: {v}"))
            })
        })
        .collect()
}

fn layout_measure(llbc: &crate::Llbc, raw: &str, target: &str, field: &str) -> Option<u64> {
    let entries: Vec<Value> = serde_json::from_str(raw).ok()?;
    let entry = entries
        .iter()
        .find(|e| e.get("key").and_then(Value::as_str) == Some(target))
        .or_else(|| (entries.len() == 1).then(|| &entries[0]))?;
    layout_u64_resolved(llbc, entry.get("value")?.get(field)?, 0)
}

/// A layout expression (`size` / `align`). Its `Deduplicated` ids index the
/// layout-expression table; the payload of a `Constant` is a constant
/// expression, whose ids index the const table instead.
fn layout_u64_resolved(llbc: &crate::Llbc, v: &Value, depth: u8) -> Option<u64> {
    if depth > 24 {
        return None;
    }
    if let Some(n) = layout_u64_literal(v) {
        return Some(n);
    }
    if let Some(id) = v.get("Deduplicated").and_then(Value::as_u64) {
        return llbc
            .layout_scalar_body(id)
            .and_then(|body| layout_u64_resolved(llbc, body, depth + 1));
    }
    if let Some(chosen) = v.get("chosen") {
        return layout_u64_resolved(llbc, chosen, depth + 1);
    }
    if let Some(body) = v
        .get("Value")
        .and_then(Value::as_array)
        .and_then(|a| a.get(1))
    {
        return layout_u64_resolved(llbc, body, depth + 1);
    }
    if let Some(constant) = v.get("Constant") {
        return layout_const_u64(llbc, constant, depth + 1);
    }
    None
}

/// A constant expression `[kind, ty]`, inline, as `{"Value": [id, [kind,
/// ty]]}`, or as `{"Deduplicated": id}` in the const table.
fn layout_const_u64(llbc: &crate::Llbc, v: &Value, depth: u8) -> Option<u64> {
    if depth > 24 {
        return None;
    }
    if let Some(n) = layout_u64_literal(v) {
        return Some(n);
    }
    if let Some(id) = v.get("Deduplicated").and_then(Value::as_u64) {
        return llbc
            .dedup_const_body(id)
            .and_then(|body| layout_const_u64(llbc, body, depth + 1));
    }
    if let Some(body) = v
        .get("Value")
        .and_then(Value::as_array)
        .and_then(|a| a.get(1))
    {
        return layout_const_u64(llbc, body, depth + 1);
    }
    v.as_array()
        .and_then(|a| a.first())
        .and_then(|kind| layout_const_u64(llbc, kind, depth + 1))
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

    /// The tag of every variant of a tagged enum none of whose variants
    /// has a field, in variant order; `None` for any other layout (a
    /// struct or single-variant type has no `Branch` discriminator).
    ///
    /// A dependency's enum reaches another crate's artefact as an `Opaque`
    /// declaration that keeps only this layout, and the tags are the only
    /// place that artefact spells the variants' values.
    pub fn fieldless_enum_tags(&self) -> Option<Vec<i64>> {
        self.discriminant_offset()?;
        if self.variant_layouts.is_empty() {
            return None;
        }
        self.variant_layouts
            .iter()
            .map(|variant| {
                if variant.field_offsets.is_empty() {
                    variant.tag_i64()
                } else {
                    None
                }
            })
            .collect()
    }

    /// Byte position of the discriminant tag (`discriminator.Branch.offset`).
    /// `None` for a single-variant type or a non-`Branch` discriminator.
    pub fn discriminant_offset(&self) -> Option<u64> {
        let offset = self.discriminator.as_ref()?.get("Branch")?.get("offset")?;
        offset
            .get("chosen")
            .and_then(Value::as_u64)
            .or_else(|| offset.as_u64())
    }

    /// Rust primitive spelling of a branching enum's physical tag.
    ///
    /// Charon records this beside the tag offset as
    /// `discriminator.Branch.int_ty`.  Consumers must use both: reading a
    /// 32-bit tag as an `i64` also reads the first four payload bytes and can
    /// turn a valid variant into a switch miss.
    ///
    /// `I128`/`U128` decline. The consumers spell a tag as a field type
    /// string, and no width above a machine word has an int-bank
    /// representation there: `get_type_flag` would route `i128` through its
    /// unknown-type fallback and describe a sixteen-byte integer as an
    /// eight-byte GC reference. Declining keeps the caller on its existing
    /// no-discriminator path instead.
    pub fn discriminant_int_type(&self) -> Option<&'static str> {
        let int_ty = self.discriminator.as_ref()?.get("Branch")?.get("int_ty")?;
        let (signed, width) = if let Some(width) = int_ty.get("Signed") {
            (true, width.as_str()?)
        } else {
            (false, int_ty.get("Unsigned")?.as_str()?)
        };
        match (signed, width) {
            (true, "I8") => Some("i8"),
            (true, "I16") => Some("i16"),
            (true, "I32") => Some("i32"),
            (true, "I64") => Some("i64"),
            (false, "U8") => Some("u8"),
            (false, "U16") => Some("u16"),
            (false, "U32") => Some("u32"),
            (false, "U64") => Some("u64"),
            _ => None,
        }
    }

    /// `true` iff a `Branch` discriminator is recorded, whatever its width.
    ///
    /// The question [`Self::discriminant_int_type`] cannot answer on its own:
    /// it returns `None` both for a layout that records no discriminator at
    /// all — a single-variant enum, or a `Known` discriminator — and for one
    /// whose width has no field-type spelling. A consumer that falls back to a
    /// default tag model owes those two cases different answers.
    pub fn has_branch_discriminant(&self) -> bool {
        self.discriminator
            .as_ref()
            .and_then(|discriminator| discriminator.get("Branch"))
            .is_some()
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

/// `name` is `None` for a positional field (tuple struct / tuple variant
/// payload). Charon spells such a field `"_N"` with `is_positional: true`;
/// the name is dropped here so a positional field and a named field that
/// happens to be called `_0` stay distinct.
#[derive(Debug, Deserialize)]
#[serde(from = "RawFieldDecl")]
pub struct FieldDecl {
    pub name: Option<String>,
    pub ty: TyRef,
    pub attr_info: Option<AttrInfo>,
}

#[derive(Deserialize)]
struct RawFieldDecl {
    name: Option<String>,
    #[serde(default)]
    is_positional: bool,
    ty: TyRef,
    #[serde(default)]
    attr_info: Option<AttrInfo>,
}

impl From<RawFieldDecl> for FieldDecl {
    fn from(raw: RawFieldDecl) -> Self {
        FieldDecl {
            name: if raw.is_positional { None } else { raw.name },
            ty: raw.ty,
            attr_info: raw.attr_info,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct VariantDecl {
    pub name: String,
    #[serde(default)]
    pub fields: Vec<FieldDecl>,
    /// Charon-assigned discriminant, kept raw because its scalar width
    /// varies by enum (`{"Unsigned":["U8","128"]}` for `Instruction`,
    /// `{"Signed":["Isize","0"]}` for others).
    /// Read via [`VariantDecl::discriminant_i64`]; staying [`Value`]
    /// keeps deserialization total under the schema-drift policy.
    #[serde(default)]
    pub discriminant: Option<Value>,
}

impl VariantDecl {
    /// Parse the discriminant to `i64`.
    ///
    /// Charon emits `{"Signed"|"Unsigned":[width, decimal_string]}`.
    /// A bare integer is the same value. Returns `None` for an absent or
    /// unparseable discriminant rather than failing — callers that need
    /// the value for an enum known to carry integer discriminants assert
    /// presence at the use site.
    pub fn discriminant_i64(&self) -> Option<i64> {
        let value = self.discriminant.as_ref()?;
        if let Some(n) = value.as_i64() {
            return Some(n);
        }
        let scalar = value.get("Scalar").unwrap_or(value);
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
    /// `methods[i].skip_binder.name` is the method a `CallKind::Trait`
    /// payload names by index. The payload no longer carries a fun-decl id.
    #[serde(default)]
    pub methods: Vec<Value>,
}

#[derive(Debug, Deserialize)]
pub struct ItemMeta {
    pub name: Vec<NameSeg>,
    pub span: SpanRef,
    pub source_text: Option<String>,
    pub attr_info: AttrInfo,
    #[serde(default)]
    pub is_local: bool,
}

impl ItemMeta {
    /// The `TraitImplId` of the trait impl block this item lives in, or
    /// `None` for an item that is not in one.
    ///
    /// [`Self::name_path`] renders that segment `"<Impl>"` and drops the
    /// id, so every trait impl in one module flattens onto a single
    /// spelling: `pyre_object::functional::<Impl>::DESCRIPTOR` is one path
    /// standing for ten distinct types' associated consts.  The id is what
    /// tells them apart, and it is the item's identity rather than a label
    /// for it — `rpython/annotator/bookkeeper.py` resolves a prebuilt
    /// through `Constant(pyobj)` (identity, per `rpython/tool/uid.py`
    /// `Hashable`) and never through a qualified name, and
    /// `rpython/rtyper/rclass.py` caches one prebuilt structure per object
    /// in an `identity_dict`.  A consumer that must distinguish siblings
    /// reads this; `name_path` stays the human-facing label.
    ///
    /// The last such segment wins: an item nested inside a trait impl
    /// belongs to the innermost one.
    pub fn trait_impl_id(&self) -> Option<u64> {
        self.name.iter().rev().find_map(|seg| match seg {
            NameSeg::Other(v) => v.get("Impl")?.get("Trait")?.as_u64(),
            NameSeg::Ident { .. } => None,
        })
    }

    /// `"crate::module::item"`-style flattened name. Trait-impl
    /// segments and other non-ident segments are rendered as
    /// `"<Variant>"` — a label, not an identity: the rendering is not
    /// injective, and [`Self::trait_impl_id`] is what recovers what it
    /// dropped.
    ///
    /// This is the template path: an `Instantiated` segment (a
    /// monomorphized copy's generic arguments) is left out, so every
    /// instance of one generic item spells the same path, the way every
    /// specialization of one `FunctionDesc` answers to `desc.name`
    /// (`rpython/annotator/description.py`).  [`Self::instantiation`]
    /// returns the arguments that tell the instances apart.
    pub fn name_path(&self) -> String {
        let mut out = String::new();
        for seg in self.template_name() {
            if !out.is_empty() {
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
                    if let Some(label) = builtin_path_label(v) {
                        out.push_str(&label);
                        continue;
                    }
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

    /// The name segments of the template path: [`Self::name`] without the
    /// `Instantiated` segment.  A leaf read takes `.last()` of this, not of
    /// `name`, whose last segment is the instantiation on a monomorphized
    /// item.
    pub fn template_name(&self) -> impl DoubleEndedIterator<Item = &NameSeg> {
        self.name
            .iter()
            .filter(|seg| !matches!(seg, NameSeg::Other(v) if instantiated_args(v).is_some()))
    }

    /// The generic arguments of a monomorphized item: the `skip_binder` of
    /// its `Instantiated` name segment, spelled like any `GenericArgs`
    /// (`regions`, `types`, `const_generics`, `trait_refs`).  `None` for an
    /// item Charon did not instantiate.
    ///
    /// This is the specialization key `FunctionDesc.cachedgraph(key)`
    /// indexes by; [`Self::name_path`] is the desc's name.
    pub fn instantiation(&self) -> Option<&Value> {
        self.name.iter().rev().find_map(|seg| match seg {
            NameSeg::Other(v) => instantiated_args(v),
            NameSeg::Ident { .. } => None,
        })
    }
}

/// The `skip_binder` generic arguments of a `PathElem::Instantiated`
/// segment, spelled `{"Instantiated": {"params": .., "skip_binder": ..}}`.
fn instantiated_args(seg: &Value) -> Option<&Value> {
    seg.as_object()?.get("Instantiated")?.get("skip_binder")
}

/// Whether a struct-root leaf names a closure env — the bare `closure`
/// or a disambiguated `closure#N` (see [`ItemMeta::name_path`]).  Leaf
/// checks that special-case closures must accept both spellings.
pub fn is_closure_leaf(leaf: &str) -> bool {
    leaf == "closure" || leaf.starts_with("closure#")
}

/// The segment label of a `PathElem::Builtin(kind, n)`, spelled
/// `{"Builtin": [kind, n]}`.
///
/// - `Closure` renders as `closure` / `closure#N`, the leaf the field
///   registry keys.
/// - `DropGlue` renders as `drop_in_place`, the method of the drop-glue
///   impl.
/// - `VTable` renders as `{vtable}`, the leaf of a trait's vtable struct.
///
/// Other builtins stay on the `<Builtin>` label.
pub fn builtin_path_label(seg: &Value) -> Option<String> {
    let arr = seg.as_object()?.get("Builtin")?.as_array()?;
    match arr.first().and_then(Value::as_str)? {
        "Closure" => {
            let n = arr.get(1).and_then(Value::as_u64).unwrap_or(0);
            Some(if n == 0 {
                "closure".to_string()
            } else {
                format!("closure#{n}")
            })
        }
        "DropGlue" => Some("drop_in_place".to_string()),
        "VTable" => Some("{vtable}".to_string()),
        _ => None,
    }
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

/// A span field. Inline bodies carry [`SpanData`]; `{"Deduplicated": id}`
/// names a row in [`crate::Llbc`]'s span table. A missing id stays unresolved.
#[derive(Debug, Clone)]
pub enum SpanRef {
    Inline(SpanData),
    Deduplicated(u64),
}

impl<'de> Deserialize<'de> for SpanRef {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        span_ref_from_value(&value)
            .ok_or_else(|| serde::de::Error::custom(format!("span is not a span body: {value}")))
    }
}

fn span_ref_from_value(value: &Value) -> Option<SpanRef> {
    if let Some(data) = value.get("data") {
        return serde_json::from_value(data.clone())
            .ok()
            .map(SpanRef::Inline);
    }
    if let Some(inner) = value.get("Untagged") {
        return span_ref_from_value(inner);
    }
    if let Some(body) = value
        .get("Value")
        .and_then(Value::as_array)
        .and_then(|arr| arr.get(1))
    {
        return span_ref_from_value(body);
    }
    let id = value.get("Deduplicated").and_then(Value::as_u64)?;
    Some(SpanRef::Deduplicated(id))
}

/// `{"data": ...}` body of a hash-consed span. The id's other occurrences
/// are `{"Deduplicated": id}`.
pub(crate) fn span_data_from_body(raw: &RawValue) -> Option<SpanData> {
    #[derive(Deserialize)]
    struct Body {
        data: SpanData,
    }
    serde_json::from_str::<Body>(raw.get()).ok().map(|b| b.data)
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpanData {
    pub file_id: u64,
    pub beg: Loc,
    pub end: Loc,
}

#[derive(Debug, Clone, Deserialize)]
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

// Types — kept thin. The lowering driver only needs to *label* types
// for diff output, not deeply reason about them, so the type table is
// not walked here.

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum TyRef {
    /// Pointer into the global type-dedup table.
    Dedup {
        #[serde(rename = "Deduplicated")]
        id: u64,
    },
    /// Inline value with hash-cons id.
    Inline {
        #[serde(rename = "Value")]
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

// Bodies

#[derive(Debug, Deserialize)]
pub struct Unstructured {
    pub locals: Locals,
    pub body: Vec<BasicBlock>,
    pub span: SpanRef,
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
    pub span: SpanRef,
    pub ty: TyRef,
}

#[derive(Debug, Deserialize)]
pub struct BasicBlock {
    pub statements: Vec<Statement>,
    pub terminator: Terminator,
    /// First successful or failed projection of [`terminator`](Self::terminator).
    /// Later [`term`](Self::term) calls clone this value instead of parsing
    /// the raw JSON again.
    #[serde(skip)]
    term_cache: OnceLock<Result<TermKind, String>>,
}

impl BasicBlock {
    /// Project the terminator into the typed [`TermKind`] enum.
    /// Returns the raw JSON in the error if a variant is unknown so
    /// callers can decide whether to fail-loud or fall back.
    ///
    /// The projection is stored on the block. Every later call returns
    /// the same `Ok` value or the same `Err` string.
    pub fn term(&self, llbc: &crate::Llbc) -> Result<TermKind, String> {
        self.term_cached(llbc).clone()
    }

    /// Borrow the cached [`term`](Self::term) projection.
    pub fn term_ref(&self, llbc: &crate::Llbc) -> Result<&TermKind, &str> {
        match self.term_cached(llbc) {
            Ok(kind) => Ok(kind),
            Err(err) => Err(err.as_str()),
        }
    }

    /// Replace the raw terminator kind, forgetting the projection of the
    /// old one so the next [`term`](Self::term) reads the new kind.
    pub fn set_terminator_kind(&mut self, kind: Value) {
        self.terminator.kind =
            serde_json::value::to_raw_value(&kind).expect("a JSON value serializes");
        self.terminator.value_cache = OnceLock::from(kind);
        self.term_cache = OnceLock::new();
    }

    fn term_cached(&self, llbc: &crate::Llbc) -> &Result<TermKind, String> {
        self.term_cache
            .get_or_init(|| decode_term_kind(&self.terminator.kind, llbc))
    }
}

/// A block's terminator, carrying its own span the way a [`Statement`]
/// does.  A call is a terminator rather than a statement, so the block's
/// last statement stands on a different line -- usually one that runs
/// after the call rather than at it.
#[derive(Debug, Deserialize)]
pub struct Terminator {
    /// Raw terminator-kind JSON text. Project to [`TermKind`] via
    /// [`BasicBlock::term`] so a parse error on a single terminator does
    /// not poison the whole function; [`Terminator::kind_value`] is the
    /// untyped tree.
    pub kind: Box<RawValue>,
    /// Optional for the same reason `kind` is projected rather than typed:
    /// a terminator missing its span costs the caller a fallback, and
    /// should not cost the body its parse. Charon writes one on every
    /// terminator, so this is `Some` for an artefact it produced.
    #[serde(default)]
    pub span: Option<SpanRef>,
    /// [`Terminator::kind_value`], parsed on first query.
    #[serde(skip)]
    value_cache: OnceLock<Value>,
}

impl Terminator {
    /// The terminator kind as an untyped JSON tree, parsed on first query.
    pub fn kind_value(&self) -> &Value {
        self.value_cache.get_or_init(|| {
            serde_json::from_str(self.kind.get()).expect("raw kind is a parsed JSON value")
        })
    }
}

#[derive(Debug, Deserialize)]
pub struct Statement {
    /// Raw statement-kind JSON text. [`Statement::stmt_kind`] parses the
    /// typed kind from it; [`Statement::kind_value`] is the untyped tree.
    pub kind: Box<RawValue>,
    pub span: SpanRef,
    /// First successful or failed projection of [`kind`](Self::kind).
    /// Later [`stmt_kind`](Self::stmt_kind) calls clone this value instead
    /// of parsing the raw JSON again.
    #[serde(skip)]
    stmt_cache: OnceLock<Result<StmtKind, String>>,
    /// [`Statement::kind_value`], parsed on first query.
    #[serde(skip)]
    value_cache: OnceLock<Value>,
}

impl Statement {
    /// The statement kind as an untyped JSON tree, parsed on first query.
    pub fn kind_value(&self) -> &Value {
        self.value_cache.get_or_init(|| {
            serde_json::from_str(self.kind.get()).expect("raw kind is a parsed JSON value")
        })
    }

    /// Project to the typed [`StmtKind`] enum.
    ///
    /// The projection is stored on the statement. Every later call returns
    /// the same `Ok` value or the same `Err` string.
    pub fn stmt_kind(&self) -> Result<StmtKind, String> {
        self.stmt_cached().clone()
    }

    /// Borrow the cached [`stmt_kind`](Self::stmt_kind) projection.
    pub fn stmt_kind_ref(&self) -> Result<&StmtKind, &str> {
        match self.stmt_cached() {
            Ok(kind) => Ok(kind),
            Err(err) => Err(err.as_str()),
        }
    }

    fn stmt_cached(&self) -> &Result<StmtKind, String> {
        self.stmt_cache.get_or_init(|| {
            let kind = self.kind.get();
            serde_json::from_str::<StmtKind>(kind).map_err(|e| format!("{e}; raw kind: {kind}"))
        })
    }
}

// Statements

#[derive(Debug, Clone, Deserialize)]
pub enum StmtKind {
    /// Local enters scope.
    StorageLive(u64),
    /// Local leaves scope.
    StorageDead(u64),
    /// `place := rvalue`
    Assign(Place, Rvalue),
    /// Borrow-checker fact. No runtime effect.
    Borrowck(Value),
    /// `Assert { cond, expected, check_kind }` — inline assertion;
    /// failure terminator is the *terminator-level* `Assert` instead.
    Assert(AssertStmt),
    /// `let _ = place` style references (MIR `PlaceMention`).
    PlaceMention(Place),
    /// Anything else (e.g. `Deinit`, `SetDiscriminant`, …).
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AssertStmt {
    pub cond: Operand,
    pub expected: bool,
    pub check_kind: Value,
}

// Places, operands, rvalues

#[derive(Debug, Clone, Deserialize)]
pub struct Place {
    pub kind: PlaceKind,
    pub ty: TyRef,
}

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
pub enum Rvalue {
    /// Second value is `WithRetag` (`"Yes"` / `"No"`).
    Use(Operand, Value),
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
    /// `Repeat(operand, elem_ty, count, trait_info)` for `[v; N]` literals.
    /// The last value is the `Copy`/`Clone` witness Charon now records.
    Repeat(Operand, TyRef, Value, Value),
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

#[derive(Debug, Clone, Deserialize)]
pub enum Operand {
    Copy(Place),
    Move(Place),
    Const(Value),
}

/// Regular `Fun` id carried by an `Operand::Const` whose kind is `FnDef`
/// (`[{"FnDef": {"kind": {"Fun": id}, ..}}, ty]`). Literals, `VTableRef`,
/// and `TraitConst` are not function items and return `None`.
pub fn const_fn_def_regular_id(llbc: &crate::Llbc, value: &Value) -> Option<u64> {
    llbc.const_expr_kind(value)?
        .get("FnDef")?
        .get("kind")?
        .get("Fun")?
        .as_u64()
}

// Terminators

#[derive(Debug, Clone, Deserialize)]
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
        /// Value whose destructor runs.  The GC liveness pass needs the place
        /// so a RootScope drop can end the bracket it opened.
        place: Place,
        /// Drop glue resolved by Charon for `place`'s type.
        fn_ptr: RegularCall,
        target: u64,
        on_unwind: u64,
    },
    #[serde(other)]
    Unknown,
}

fn decode_term_kind(kind: &RawValue, llbc: &crate::Llbc) -> Result<TermKind, String> {
    let raw = kind.get();
    // Only a `Switch` can carry `data`, whose arm constants are read off
    // the dedup tables; every other kind parses straight from the text.
    let is_switch = raw
        .trim_start()
        .strip_prefix('{')
        .is_some_and(|rest| rest.trim_start().starts_with("\"Switch\""));
    let mut term = if is_switch {
        let kind: Value = serde_json::from_str(raw).map_err(|e| format!("{e}; raw kind: {raw}"))?;
        if let Some(sw) = kind.get("Switch")
            && sw.get("data").is_some()
        {
            return decode_switch(sw, llbc).map_err(|e| format!("{e}; raw kind: {kind}"));
        }
        TermKind::deserialize(&kind).map_err(|e| format!("{e}; raw kind: {kind}"))?
    } else {
        serde_json::from_str(raw).map_err(|e| format!("{e}; raw kind: {raw}"))?
    };
    match &mut term {
        TermKind::Call {
            call:
                CallPayload {
                    func: CallFunc::Regular(reg),
                    ..
                },
            ..
        } => reg.take_instance_arguments(llbc),
        TermKind::Drop { fn_ptr, .. } => fn_ptr.take_instance_arguments(llbc),
        _ => {}
    }
    Ok(term)
}

/// `Switch { data: {scrutinee, branches, fallback}, branches }` decodes
/// straight into [`TermKind::Switch`]. Arm constants may be
/// `{"Deduplicated": id}` and are read from [`crate::Llbc::dedup_const_body`].
fn decode_switch(sw: &Value, llbc: &crate::Llbc) -> Result<TermKind, String> {
    let data = sw.get("data").ok_or("switch missing data")?;
    let bbs = sw
        .get("branches")
        .and_then(Value::as_array)
        .ok_or("switch missing block branches")?;
    let discr_v = data
        .get("scrutinee")
        .and_then(|s| s.get("Value"))
        .ok_or("switch missing scrutinee")?;
    let discr = Operand::deserialize(discr_v).map_err(|e| e.to_string())?;
    let arms = data
        .get("branches")
        .and_then(Value::as_array)
        .ok_or("switch missing arms")?;
    let fallback = data.get("fallback").and_then(Value::as_u64);
    let bb_of = |id: u64| bbs.get(id as usize).and_then(Value::as_u64);
    let mut decoded: Vec<(Value, u64, Option<bool>)> = Vec::new();
    for arm in arms {
        let pair = arm.as_array().ok_or("switch arm is not a pair")?;
        let target = pair
            .get(1)
            .and_then(Value::as_u64)
            .and_then(bb_of)
            .ok_or("switch arm target")?;
        let lit = llbc
            .const_expr_literal(pair.first().ok_or("switch arm const")?)
            .ok_or("switch arm const unresolved")?;
        let flag = lit.get("Bool").and_then(Value::as_bool);
        decoded.push((lit, target, flag));
    }
    let all_bool = !decoded.is_empty() && decoded.iter().all(|(_, _, flag)| flag.is_some());
    let targets = if all_bool {
        let mut then_bb = fallback.and_then(bb_of);
        let mut else_bb = None;
        for (_, target, flag) in &decoded {
            if flag == &Some(true) {
                then_bb = Some(*target);
            } else {
                else_bb = Some(*target);
            }
        }
        let then_bb = then_bb.ok_or("bool switch missing then")?;
        let else_bb = else_bb
            .or_else(|| fallback.and_then(bb_of))
            .ok_or("bool switch missing else")?;
        SwitchTargets::If(then_bb, else_bb)
    } else {
        let default = fallback.and_then(bb_of).ok_or("switch missing fallback")?;
        let arms = decoded
            .into_iter()
            .map(|(scalar, target, _)| (scalar, target))
            .collect();
        SwitchTargets::SwitchInt(Value::Null, arms, default)
    };
    Ok(TermKind::Switch { discr, targets })
}

#[derive(Debug, Clone, Deserialize)]
pub enum SwitchTargets {
    /// Boolean switch: `[then_bb, else_bb]`.
    If(u64, u64),
    /// `SwitchInt(int_ty, [(scalar, bb)], default_bb)`.
    SwitchInt(Value, Vec<(Value, u64)>, u64),
}

#[derive(Debug, Clone, Deserialize)]
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
#[derive(Debug, Clone, Deserialize)]
pub enum CallFunc {
    Regular(RegularCall),
    /// `dyn Trait` virtual call. The operand carries the fat pointer
    /// the dispatch reads from.
    Dynamic(Operand),
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RegularCall {
    pub kind: CallKind,
    pub generics: Value,
}

impl RegularCall {
    /// Spell a call to a monomorphized copy the way a call to its generic
    /// item is spelled: the callee's instance arguments become the call's
    /// `generics`.  Charon's `--monomorphize` moves the call site's type,
    /// const and trait arguments into the callee's `Instantiated` name
    /// segment and leaves the call with its regions only, so a reader of
    /// `generics` would see an unparameterized call.  The call keeps its own
    /// regions; a call that already carries arguments is left alone.
    fn take_instance_arguments(&mut self, llbc: &crate::Llbc) {
        let CallKind::Fun(FunId::Regular { id }) = &self.kind else {
            return;
        };
        let Some(args) = llbc
            .fn_by_id(*id)
            .and_then(|fd| fd.item_meta.instantiation())
        else {
            return;
        };
        let Some(generics) = self.generics.as_object_mut() else {
            return;
        };
        const KEYS: [&str; 3] = ["types", "const_generics", "trait_refs"];
        let unparameterized = KEYS.iter().all(|key| {
            generics
                .get(*key)
                .and_then(Value::as_array)
                .is_none_or(Vec::is_empty)
        });
        if !unparameterized {
            return;
        }
        for key in KEYS {
            if let Some(v) = args.get(key) {
                generics.insert(key.to_string(), v.clone());
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone)]
pub enum FunId {
    /// `CallKind::Fun` carries the fun-decl id as a bare integer.
    Regular {
        id: u64,
    },
    Other(Value),
}

impl<'de> Deserialize<'de> for FunId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        if let Some(id) = value.as_u64() {
            return Ok(FunId::Regular { id });
        }
        Ok(FunId::Other(value))
    }
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

    fn item_meta(name: Value) -> ItemMeta {
        serde_json::from_value(serde_json::json!({
            "name": name,
            "span": {"Deduplicated": 0},
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": true
        }))
        .unwrap()
    }

    /// A monomorphized copy keeps its generic item's path; its arguments
    /// come from `instantiation`.
    #[test]
    fn instantiated_segment_is_left_out_of_the_template_path() {
        let args = serde_json::json!({
            "regions": [], "types": [{"Deduplicated": 3}],
            "const_generics": [], "trait_refs": []
        });
        let mono = item_meta(serde_json::json!([
            {"Ident": ["core", 0]},
            {"Ident": ["ptr", 0]},
            {"Ident": ["null", 0]},
            {"Instantiated": {"params": {}, "skip_binder": args.clone(), "kind": "Other"}}
        ]));
        assert_eq!(mono.name_path(), "core::ptr::null");
        assert_eq!(mono.instantiation(), Some(&args));
        assert!(matches!(
            mono.template_name().last(),
            Some(NameSeg::Ident { ident: (leaf, _) }) if leaf == "null"
        ));

        let generic = item_meta(serde_json::json!([
            {"Ident": ["core", 0]},
            {"Ident": ["ptr", 0]},
            {"Ident": ["null", 0]}
        ]));
        assert_eq!(generic.name_path(), "core::ptr::null");
        assert_eq!(generic.instantiation(), None);
    }

    /// A call to a monomorphized copy reads the copy's instance arguments
    /// as its `generics`; a call that carries its own keeps them.
    #[test]
    fn a_call_to_an_instance_carries_the_instance_arguments() {
        let doc = r#"{"charon_version":"t","has_errors":false,
            "translated":{"crate_name":"c","fun_decls":[{
                "def_id":0,
                "item_meta":{"name":[{"Ident":["core",0]},{"Ident":["into",0]},
                    {"Instantiated":{"params":{},"kind":"Other","skip_binder":{
                        "regions":[],"types":[{"Deduplicated":3},{"Deduplicated":4}],
                        "const_generics":[],"trait_refs":[]}}}],
                    "span":{"Deduplicated":0},"source_text":null,
                    "attr_info":{"attributes":[],"inline":null,"rename":null,"public":true},
                    "is_local":false},
                "signature":{"is_unsafe":false,"inputs":[],"output":{"Deduplicated":3}},
                "body":null}]}}"#;
        let llbc = crate::Llbc::from_slice(doc.as_bytes()).expect("fixture parses");
        let call = |generics: Value| {
            serde_json::json!({"Call": {
                "call": {
                    "func": {"Regular": {"kind": {"Fun": 0}, "generics": generics}},
                    "args": [],
                    "dest": {"kind": {"Local": 0}, "ty": {"Deduplicated": 3}}
                },
                "target": 1,
                "on_unwind": 2
            }})
        };
        let generics_of = |kind: &Value| match decode_term_kind(
            &serde_json::value::to_raw_value(kind).expect("a JSON value serializes"),
            &llbc,
        ) {
            Ok(TermKind::Call {
                call:
                    CallPayload {
                        func: CallFunc::Regular(reg),
                        ..
                    },
                ..
            }) => reg.generics,
            other => panic!("not a regular call: {other:?}"),
        };
        let mono = generics_of(&call(serde_json::json!({
            "regions": [{"Body": 9}], "types": [], "const_generics": [], "trait_refs": []
        })));
        assert_eq!(
            mono,
            serde_json::json!({
                "regions": [{"Body": 9}],
                "types": [{"Deduplicated": 3}, {"Deduplicated": 4}],
                "const_generics": [], "trait_refs": []
            })
        );
        let own = serde_json::json!({
            "regions": [], "types": [{"Deduplicated": 5}], "const_generics": [], "trait_refs": []
        });
        assert_eq!(generics_of(&call(own.clone())), own);
    }

    /// A positional field loses its `_N` spelling; a named `_0` keeps it.
    #[test]
    fn positional_field_has_no_name() {
        let ty = r#"{"Deduplicated": 0}"#;
        let positional: FieldDecl = serde_json::from_str(&format!(
            r#"{{"name": "_0", "is_positional": true, "ty": {ty}}}"#
        ))
        .unwrap();
        assert_eq!(positional.name, None);
        let named: FieldDecl = serde_json::from_str(&format!(
            r#"{{"name": "_0", "is_positional": false, "ty": {ty}}}"#
        ))
        .unwrap();
        assert_eq!(named.name.as_deref(), Some("_0"));
    }

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
        assert!(!layout.repr.as_ref().unwrap().transparent);
        // single-variant type has no decodable discriminant tag
        assert_eq!(layout.discriminant_offset(), None);
    }

    #[test]
    fn transparent_repr_is_preserved() {
        let json = r#"{
            "size": 8, "align": 8,
            "variant_layouts": [{"field_offsets": [0]}],
            "repr": {"repr_algo": "Rust", "transparent": true}
        }"#;
        let layout: TypeLayout = serde_json::from_str(json).unwrap();
        assert!(layout.repr.unwrap().transparent);
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
        assert_eq!(layout.discriminant_int_type(), Some("u8"));
    }

    /// A fieldless enum's variant values are the tags its layout writes;
    /// a struct, or an enum with a field, has none.
    #[test]
    fn fieldless_enum_tags_come_from_the_taggers() {
        let json = r#"{
            "size": 1, "align": 1,
            "discriminator": {"Branch": {"offset": 0, "int_ty": {"Signed": "I8"}}},
            "variant_layouts": [
                {"field_offsets": [], "tagger": [[0, {"Signed": ["I8", "-1"]}]]},
                {"field_offsets": [], "tagger": [[0, {"Signed": ["I8", "4"]}]]}
            ]
        }"#;
        let layout: TypeLayout = serde_json::from_str(json).unwrap();
        assert_eq!(layout.fieldless_enum_tags(), Some(vec![-1, 4]));

        let unit_struct: TypeLayout =
            serde_json::from_str(r#"{"size": 0, "variant_layouts": [{"field_offsets": []}]}"#)
                .unwrap();
        assert_eq!(unit_struct.fieldless_enum_tags(), None);

        let payload = r#"{
            "size": 16,
            "discriminator": {"Branch": {"offset": 0, "int_ty": {"Unsigned": "U8"}}},
            "variant_layouts": [
                {"field_offsets": [], "tagger": [[0, {"Unsigned": ["U8", "0"]}]]},
                {"field_offsets": [8], "tagger": [[0, {"Unsigned": ["U8", "1"]}]]}
            ]
        }"#;
        let payload: TypeLayout = serde_json::from_str(payload).unwrap();
        assert_eq!(payload.fieldless_enum_tags(), None);
    }

    #[test]
    fn enum_discriminant_signedness_and_width_are_preserved() {
        for (int_ty, expected) in [
            (r#"{"Signed":"I16"}"#, Some("i16")),
            (r#"{"Signed":"I32"}"#, Some("i32")),
            (r#"{"Unsigned":"U64"}"#, Some("u64")),
            // Wider than a machine word: no int-bank field type spells it.
            (r#"{"Signed":"I128"}"#, None),
            (r#"{"Unsigned":"U128"}"#, None),
        ] {
            let json = format!(
                r#"{{
                    "discriminator": {{"Branch": {{"offset": 4, "int_ty": {int_ty}}}}},
                    "variant_layouts": [{{"field_offsets": []}}, {{"field_offsets": []}}]
                }}"#
            );
            let layout: TypeLayout = serde_json::from_str(&json).unwrap();
            assert_eq!(layout.discriminant_int_type(), expected);
            // A declined width is still a recorded tag: the two `None`s a
            // consumer sees from `discriminant_int_type` are distinguishable.
            assert!(layout.has_branch_discriminant());
        }
        let no_discriminator: TypeLayout =
            serde_json::from_str(r#"{"variant_layouts": [{"field_offsets": []}]}"#).unwrap();
        assert!(!no_discriminator.has_branch_discriminant());
        assert_eq!(no_discriminator.discriminant_int_type(), None);
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
