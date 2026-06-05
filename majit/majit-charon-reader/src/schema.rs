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
    /// `trait_impls`, `ordered_decls`, `files`, `options`,
    /// `target_information`, `item_names`, `assoc_item_names`,
    /// `short_names` — kept opaque until a driver pass needs them.
    /// Charon's release-to-release renames of these surfaces are the
    /// most common source of schema drift; staying opaque here means
    /// a Charon bump does not require recompiling this crate.
    #[serde(flatten)]
    pub rest: std::collections::BTreeMap<String, Value>,
}
