//! Stable-Rust parser for Charon `.llbc` / `.ullbc` JSON artefacts.
//!
//! This crate is the input layer of issue #97's Step 3 MIR-driven
//! flowspace driver. It exposes:
//!
//!   - [`schema`] — `serde::Deserialize` structs covering the subset of
//!     Charon's IR we actually consume. Schema fields we do not yet
//!     consume are kept as opaque [`serde_json::Value`] so that newer
//!     Charon versions stay round-trippable; we widen the typed schema
//!     incrementally as Step 3 needs each piece.
//!   - [`Llbc`] — a thin wrapper around [`schema::LlbcFile`] with
//!     lookup helpers (`local_fn`, `iter_local_fns`, etc.).
//!   - [`SchemaError`] — fail-loud error type. The crate never silently
//!     drops bodies; an unrecognised variant returns a hard error.
//!
//! The crate compiles on **stable Rust**. The pinned-nightly toolchain
//! required to produce `.llbc` lives inside Charon itself
//! (`scripts/install-charon.sh`); nothing in this crate touches it.

#![forbid(unsafe_code)]

pub mod schema;
pub mod ullbc;

pub use schema::LlbcFile;
pub use ullbc::{
    BasicBlock, FieldDecl, FunDecl, GlobalDecl, Locals, Statement, StmtKind, TermKind, TraitDecl,
    TypeDecl, TypeDeclKind, Unstructured, VariantDecl,
};

use std::path::Path;

/// Loaded `.llbc` / `.ullbc` artefact + lookup helpers.
#[derive(Debug)]
pub struct Llbc {
    pub file: LlbcFile,
}

impl Llbc {
    /// Load and parse a `.llbc` / `.ullbc` JSON file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SchemaError> {
        let bytes = std::fs::read(path.as_ref()).map_err(SchemaError::Io)?;
        Self::from_slice(&bytes)
    }

    /// Parse a `.llbc` / `.ullbc` artefact from an in-memory byte slice.
    pub fn from_slice(bytes: &[u8]) -> Result<Self, SchemaError> {
        let file: LlbcFile = serde_json::from_slice(bytes).map_err(SchemaError::Parse)?;
        Ok(Self { file })
    }

    /// Look up a local-crate function whose name ends with `::<name>`.
    pub fn local_fn(&self, name: &str) -> Option<&FunDecl> {
        let suffix = format!("::{name}");
        for f in self.iter_local_fns() {
            let path = f.item_meta.name_path();
            if path == name || path.ends_with(&suffix) {
                return Some(f);
            }
        }
        None
    }

    /// Look up a `FunDecl` by its Charon `def_id`. The `fun_decls`
    /// array is indexed by `def_id` (verified against extracted
    /// corpora), so this is an O(1) bounds-checked lookup.
    pub fn fn_by_id(&self, def_id: u64) -> Option<&FunDecl> {
        self.file
            .translated
            .fun_decls
            .get(def_id as usize)?
            .as_ref()
    }

    /// Look up a `GlobalDecl` by its Charon `def_id`. Same indexing
    /// invariant as [`fn_by_id`].
    pub fn global_by_id(&self, def_id: u64) -> Option<&GlobalDecl> {
        self.file
            .translated
            .global_decls
            .get(def_id as usize)?
            .as_ref()
    }

    /// Look up a `TypeDecl` by its Charon `def_id`. Same indexing
    /// invariant as [`fn_by_id`].
    pub fn type_by_id(&self, def_id: u64) -> Option<&TypeDecl> {
        self.file
            .translated
            .type_decls
            .get(def_id as usize)?
            .as_ref()
    }

    /// Look up a `TraitDecl` by its Charon `def_id`.
    pub fn trait_by_id(&self, def_id: u64) -> Option<&TraitDecl> {
        self.file
            .translated
            .trait_decls
            .get(def_id as usize)?
            .as_ref()
    }

    /// Iterate over every present `TypeDecl`.
    pub fn iter_type_decls(&self) -> impl Iterator<Item = &TypeDecl> {
        self.file
            .translated
            .type_decls
            .iter()
            .filter_map(Option::as_ref)
    }

    /// Iterate over every present `TraitDecl`.
    pub fn iter_trait_decls(&self) -> impl Iterator<Item = &TraitDecl> {
        self.file
            .translated
            .trait_decls
            .iter()
            .filter_map(Option::as_ref)
    }

    /// Iterate over every present `FunDecl` (skipping opaque `null` entries).
    pub fn iter_local_fns(&self) -> impl Iterator<Item = &FunDecl> {
        self.file
            .translated
            .fun_decls
            .iter()
            .filter_map(Option::as_ref)
    }

    /// Crate name (the `crate_name` field from `.llbc.translated`).
    pub fn crate_name(&self) -> &str {
        &self.file.translated.crate_name
    }
}

/// Errors produced when loading / parsing a `.llbc` artefact.
#[derive(Debug)]
pub enum SchemaError {
    Io(std::io::Error),
    Parse(serde_json::Error),
    Decode(String),
}

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchemaError::Io(e) => write!(f, "io: {e}"),
            SchemaError::Parse(e) => write!(f, "parse: {e}"),
            SchemaError::Decode(s) => write!(f, "decode: {s}"),
        }
    }
}

impl std::error::Error for SchemaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SchemaError::Io(e) => Some(e),
            SchemaError::Parse(e) => Some(e),
            SchemaError::Decode(_) => None,
        }
    }
}
