//! Top-level Charon `.llbc` schema.
//!
//! Only the fields the lowering driver actually reads are typed;
//! the rest stay as `serde_json::Value` so that newer Charon versions
//! load without code changes.

use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub struct LlbcFile {
    pub charon_version: String,
    pub has_errors: bool,
    pub translated: Translated,
}

#[derive(Debug, Deserialize)]
pub struct Translated {
    pub crate_name: String,
    /// Extraction target metadata. Charon serializes this as an ordered
    /// sequence because one artefact can in principle name several targets;
    /// preserving that shape lets consumers require one unambiguous pointer
    /// width instead of silently choosing a map winner.
    #[serde(default)]
    pub target_information: Vec<TargetInformationEntry>,
    pub fun_decls: Vec<Option<crate::ullbc::FunDecl>>,
    /// Static / const items the MIR references via `Place::Global` and
    /// `Operand::Const(Global { ... })`. Indexed by `def_id` (the same
    /// invariant `fun_decls` upholds; verified against extracted
    /// corpora).
    #[serde(default)]
    pub global_decls: Vec<Option<crate::ullbc::GlobalDecl>>,
    /// User-defined types (`struct` / `enum` / alias / opaque).
    /// Indexed by `def_id`. Consumed to populate
    /// `SemanticProgram.{known_struct_names, struct_fields,
    /// immutable_fields}`.
    #[serde(default)]
    pub type_decls: Vec<Option<crate::ullbc::TypeDecl>>,
    /// Trait declarations. Indexed by `def_id`. Consumed for
    /// `SemanticProgram.known_trait_names`.
    #[serde(default)]
    pub trait_decls: Vec<Option<crate::ullbc::TraitDecl>>,
    /// `impl Trait for T` table, indexed by trait-impl id. Kept as raw
    /// `Value` entries and projected on demand (see
    /// [`crate::Llbc::trait_impls_raw`]).  Read by the front-end's
    /// trait-associated-type resolution.
    ///
    /// Every other top-level surface Charon emits (`ordered_decls`,
    /// `options`, `item_names`,
    /// `assoc_item_names`, `short_names`, …) is intentionally not
    /// modelled: serde skips unknown fields without allocating, which
    /// both keeps the loader resilient to Charon's release-to-release
    /// renames *and* avoids materialising the whole document as a
    /// `serde(flatten)` catch-all (the latter forces serde to buffer
    /// the entire `translated` object into an in-memory `Content` tree).
    #[serde(default)]
    pub trait_impls: Vec<Value>,
    /// Source files, indexed by the `file_id` every
    /// [`crate::ullbc::SpanData`] carries.  Without it a span names a
    /// line in a file nothing can name.
    ///
    /// Each entry also carries the file's entire `contents`; that field
    /// is deliberately unmodelled, so serde walks past it without
    /// allocating.  The array was already being walked as an unknown
    /// field, so what this adds is the `id` and `name` of each entry and
    /// nothing else.
    #[serde(default)]
    pub files: Vec<SourceFile>,
}

/// One row of [`Translated::files`].
#[derive(Debug, Deserialize)]
pub struct SourceFile {
    pub id: u64,
    /// Charon's `FileName`, a single-variant object (`{"Local": path}`,
    /// `{"Virtual": path}`, …).  Held raw and projected by
    /// [`crate::Llbc::file_path`] so a variant this crate has never seen
    /// loads rather than failing the whole artefact.
    pub name: Value,
}

#[derive(Debug, Deserialize)]
pub struct TargetInformationEntry {
    pub key: String,
    pub value: TargetInformation,
}

#[derive(Debug, Deserialize)]
pub struct TargetInformation {
    pub target_pointer_size: u8,
    pub is_little_endian: bool,
}
